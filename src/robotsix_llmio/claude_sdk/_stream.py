"""Shared Claude Agent SDK streaming loop — used by both the no-tools
model path (``ClaudeSDKModel._invoke``) and the SDK-tools provider path
(``_SdkToolAgentHandle._invoke_query``).
"""

from __future__ import annotations

import asyncio
import contextvars
import json
import logging
from collections.abc import AsyncIterator, Callable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any, Literal

log = logging.getLogger("robotsix_llmio.claude_sdk")

ActivityKind = Literal["tool_call", "tool_result", "thinking", "text"]


@dataclass(frozen=True)
class ClaudeSDKActivityEvent:
    """One live signal from the Claude Agent SDK's streaming loop.

    Mirrors what the diagnostic log line already captures — tool calls, tool
    results, extended-thinking, and intermediate assistant text — as
    structured data a caller can forward as UI feedback (e.g. into
    robotsix-chat's SSE channel) instead of only reading it from logs.
    """

    kind: ActivityKind
    turn: int
    tool_name: str | None = None
    detail: str = ""
    is_error: bool = False


_current_on_event: contextvars.ContextVar[
    Callable[[ClaudeSDKActivityEvent], None] | None
] = contextvars.ContextVar("_current_on_event", default=None)


@contextmanager
def activity_events(
    on_event: Callable[[ClaudeSDKActivityEvent], None],
) -> Iterator[None]:
    """Route Claude Agent SDK activity events to *on_event* for this context.

    Applies to any ``claude_sdk`` call made within the ``with`` block —
    both the no-tools ``ClaudeSDKModel`` path and the SDK-tools
    ``_SdkToolAgentHandle`` path — since both funnel through
    :func:`_stream_query`, which reads this ambient callback when no
    explicit *on_event* argument is given. A context variable is used
    (rather than a constructor/call argument on ``build_agent()``/``run()``)
    so a caller doesn't need to know which transport ``build_agent()``
    resolved to, and the shared ``LLMProvider`` interface stays untouched.

    ``contextvars`` propagate into the ``asyncio.Task`` the streaming loop
    runs in (a task copies its creator's context at creation time), so this
    works across the ``await`` inside ``handle.run(...)``.
    """
    token = _current_on_event.set(on_event)
    try:
        yield
    finally:
        _current_on_event.reset(token)


def _short(value: Any, limit: int = 200) -> str:
    """One-line, length-capped repr of a tool input / text for logging."""
    s = value if isinstance(value, str) else json.dumps(value, default=str)
    s = " ".join(s.split())
    return s if len(s) <= limit else s[:limit] + "…"


def _log_stream_message(
    message: Any,
    turn: list[int],
    label: str,
    on_event: Callable[[ClaudeSDKActivityEvent], None] | None = None,
) -> None:
    """Emit a concise INFO line for one streamed Claude SDK message.

    Gives live feedback on what the agent is doing — turns, tool calls, tool
    results, the final result — even when Langfuse spans haven't flushed yet
    (a stuck agent never completes its span, so this is the only signal).
    *turn* is a 1-element list used as a mutable counter across the loop.

    When *on_event* is given, it is also called with a
    :class:`ClaudeSDKActivityEvent` for each tool call, tool result,
    thinking block, or intermediate assistant text — the same events this
    function logs, structured for a caller to forward elsewhere (e.g. a chat
    UI). Never raises: a broken callback only skips that one event.
    """
    cls = type(message).__name__
    try:
        if cls == "AssistantMessage":
            turn[0] += 1
            for block in getattr(message, "content", []) or []:
                bcls = type(block).__name__
                if bcls == "TextBlock":
                    txt = getattr(block, "text", "") or ""
                    if txt.strip():
                        log.info("%s turn %d: text — %s", label, turn[0], _short(txt))
                        _emit(
                            on_event,
                            ClaudeSDKActivityEvent(
                                kind="text", turn=turn[0], detail=_short(txt)
                            ),
                        )
                elif bcls == "ToolUseBlock":
                    tool_name = getattr(block, "name", "?")
                    tool_input = _short(getattr(block, "input", {}))
                    log.info(
                        "%s turn %d: tool_use %s(%s)",
                        label,
                        turn[0],
                        tool_name,
                        tool_input,
                    )
                    _emit(
                        on_event,
                        ClaudeSDKActivityEvent(
                            kind="tool_call",
                            turn=turn[0],
                            tool_name=tool_name,
                            detail=tool_input,
                        ),
                    )
                elif bcls == "ThinkingBlock":
                    n_chars = len(getattr(block, "thinking", "") or "")
                    log.info("%s turn %d: thinking (%d chars)", label, turn[0], n_chars)
                    _emit(
                        on_event,
                        ClaudeSDKActivityEvent(
                            kind="thinking",
                            turn=turn[0],
                            detail=f"{n_chars} chars",
                        ),
                    )
        elif cls in ("UserMessage", "ToolResultMessage"):
            for block in getattr(message, "content", []) or []:
                if type(block).__name__ == "ToolResultBlock":
                    is_err = bool(getattr(block, "is_error", False))
                    detail = _short(getattr(block, "content", ""))
                    log.info(
                        "%s tool_result%s — %s",
                        label,
                        " [ERROR]" if is_err else "",
                        detail,
                    )
                    _emit(
                        on_event,
                        ClaudeSDKActivityEvent(
                            kind="tool_result",
                            turn=turn[0],
                            detail=detail,
                            is_error=is_err,
                        ),
                    )
        elif cls == "ResultMessage":
            log.info(
                "%s result: subtype=%s is_error=%s turns=%d duration_ms=%s",
                label,
                getattr(message, "subtype", "?"),
                getattr(message, "is_error", "?"),
                turn[0],
                getattr(message, "duration_ms", "?"),
            )
    except Exception:
        log.debug(
            "_log_message: unexpected SDK message shape; diagnostic logging skipped",
            exc_info=True,
        )


def _emit(
    on_event: Callable[[ClaudeSDKActivityEvent], None] | None,
    event: ClaudeSDKActivityEvent,
) -> None:
    """Call *on_event* with *event*, swallowing any callback failure.

    A caller's forwarding callback (e.g. publishing to a UI event bus) must
    never break the SDK streaming loop — logging already happened, so the
    turn is not otherwise lost.
    """
    if on_event is None:
        return
    try:
        on_event(event)
    except Exception:
        log.debug("on_event callback raised; activity event dropped", exc_info=True)


async def _messages_aiter(
    messages: list[dict[str, Any]],
) -> AsyncIterator[dict[str, Any]]:
    """Fresh async iterator over streaming-input *messages*.

    Built per call (not shared) so a retry of the same prompt list never
    replays an already-exhausted generator.
    """
    for message in messages:
        yield message


async def _stream_query(
    prompt: str | list[dict[str, Any]],
    options: Any,  # ClaudeAgentOptions (lazy import to avoid cycle with model)
    label: str,
    *,
    extra_transient: Callable[[Exception], bool] | None = None,
    on_event: Callable[[ClaudeSDKActivityEvent], None] | None = None,
) -> tuple[str, Any, str]:
    """Run the Claude Agent SDK streaming loop under the per-call wall-clock cap.

    *prompt* is either the plain-text prompt or a list of streaming-input
    message dicts (used to attach base64 image blocks); a list is wrapped in
    a fresh async iterator here so retries can re-send it safely.

    Returns ``(text, result, reasoning)`` — joined assistant text (with the
    ``ResultMessage.result`` fallback), the captured ``ResultMessage``, and the
    joined extended-thinking (``ThinkingBlock``) content the model emitted
    before its answer (``""`` when thinking is off or unavailable). The SDK
    streams thinking as its own block type; capturing it here lets callers
    surface the model's reasoning in traces instead of discarding it.

    *on_event*, when given, is called live with a
    :class:`ClaudeSDKActivityEvent` for each tool call, tool result, thinking
    block, or intermediate assistant text streamed by the SDK — the same
    activity already logged at INFO level, structured for a caller to
    forward as UI feedback (e.g. robotsix-chat's SSE channel) instead of only
    reading it from logs.

    Converts a wall-clock timeout into :class:`ClaudeSDKQueryTimeout` (always).
    If *extra_transient* is given and returns ``True`` for a non-timeout
    exception, that exception is wrapped in :class:`ClaudeSDKTurnLimitError`.

    Raises :class:`ClaudeSDKUsageExhaustedError` when usage-exhaustion wording
    is detected, via either of two shapes the SDK can surface it as: (a) a
    completed result flagged ``is_error=True`` whose text reports exhausted
    credits — that shape does not raise upstream, so left unhandled it would
    return the error text as if it were a genuine reply; or (b) the SDK
    collapsing it into a raised exception (e.g. the degenerate-success
    message) that discards the real text — checked against whatever text had
    already streamed into ``chunks`` before that exception fired.
    """
    from claude_agent_sdk import (
        AssistantMessage,
        ResultMessage,
        TextBlock,
        query,
    )

    from ..core import constants
    from ..exceptions import RobotsixLLMIOError
    from ._errors import (
        ClaudeSDKAPIError,
        ClaudeSDKPermanentAPIError,
        ClaudeSDKQueryTimeout,
        ClaudeSDKTurnLimitError,
        ClaudeSDKUsageExhaustedError,
    )
    from .transient import is_permanent_api_error_text, is_usage_exhausted_text

    chunks: list[str] = []
    thoughts: list[str] = []
    result: Any = None
    turn = [0]
    effective_on_event = on_event if on_event is not None else _current_on_event.get()

    sdk_prompt: str | AsyncIterator[dict[str, Any]] = (
        prompt if isinstance(prompt, str) else _messages_aiter(prompt)
    )

    async def _consume() -> None:
        nonlocal result
        async for message in query(prompt=sdk_prompt, options=options):
            _log_stream_message(message, turn, label, effective_on_event)
            if isinstance(message, AssistantMessage):
                for block in message.content:
                    if isinstance(block, TextBlock):
                        chunks.append(block.text)
                    # ``ThinkingBlock`` is matched by class name rather than by
                    # import: the symbol isn't exported by every SDK version, and
                    # a name check degrades to "no reasoning captured" instead of
                    # an ImportError on older SDKs.
                    elif type(block).__name__ == "ThinkingBlock":
                        thinking = getattr(block, "thinking", "") or ""
                        if thinking.strip():
                            thoughts.append(thinking)
            elif isinstance(message, ResultMessage):
                result = message

    try:
        # Hard wall-clock cap so a stalled CLI subprocess fails fast and
        # retryable instead of hanging on the SDK's own ~2h backstop.
        await asyncio.wait_for(_consume(), timeout=constants.SDK_QUERY_TIMEOUT)
    except TimeoutError as exc:
        raise ClaudeSDKQueryTimeout(
            f"Claude Agent SDK query exceeded the "
            f"{constants.SDK_QUERY_TIMEOUT:.0f}s per-call wall-clock cap "
            f"({label}) — the call stalled without completing. "
            f"Treated as transient so the bounded retry re-runs it."
        ) from exc
    except Exception as exc:
        # claude_agent_sdk sometimes collapses a usage-exhausted frame into a
        # generic raised exception (e.g. the degenerate-success message),
        # discarding the real assistant-visible text ("You're out of usage
        # credits") before we would otherwise see it. The text already
        # streamed into `chunks` before that happened, so check it here too —
        # not just on the clean-return path below — or this cause is
        # misclassified as an ordinary transient error, retried 3x at the
        # same exhausted tier, and finally leaked to the user verbatim.
        partial_text = "".join(chunks).strip()
        if partial_text and is_usage_exhausted_text(partial_text):
            raise ClaudeSDKUsageExhaustedError(partial_text) from exc
        # Same shape, same reasoning: an API 400 streams in as assistant text
        # and is then collapsed into the degenerate-success frame, which the
        # classifier would otherwise call transient — burning all retries on a
        # request that can never succeed. Checked before that branch so the
        # permanent cause wins.
        if partial_text and is_permanent_api_error_text(partial_text):
            raise ClaudeSDKPermanentAPIError(
                f"Anthropic API rejected the request ({label}): {partial_text}"
            ) from exc
        if extra_transient is not None and extra_transient(exc):
            raise ClaudeSDKTurnLimitError(
                f"Claude Agent SDK hit the turn cap without "
                f"producing a final answer ({label}). The "
                f"agent loop did not converge — it kept taking turns instead "
                f"of terminating. This is a hard failure; retrying the "
                f"identical request would hit the cap again. SDK error: {exc}"
            ) from exc
        # Already-wrapped library errors (RobotsixLLMIOError subclasses)
        # pass through unchanged; raw claude_agent_sdk transport/process
        # failures are wrapped so callers can catch a single base class.
        if isinstance(exc, RobotsixLLMIOError):
            raise
        raise ClaudeSDKAPIError(
            f"Claude Agent SDK transport/process failure ({label}): {exc}"
        ) from exc

    text = "".join(chunks).strip()
    if not text and result is not None:
        text = (getattr(result, "result", None) or "").strip()
    reasoning = "\n\n".join(thoughts).strip()

    # The SDK sometimes reports usage-exhaustion as a normal-looking
    # completion (ResultMessage.is_error=True, no exception raised) rather
    # than surfacing it as an error — left unhandled, the assistant-visible
    # "You're out of usage credits" text would be returned as a genuine
    # reply. Scoped narrowly to this one signature: any other is_error=True
    # shape is left exactly as-is (out of scope for this fix).
    if (
        result is not None
        and getattr(result, "is_error", False)
        and is_usage_exhausted_text(text)
    ):
        raise ClaudeSDKUsageExhaustedError(text)

    # Likewise for a request-validation rejection reported as a normal-looking
    # completion — without this the "API Error: 400 ..." text is returned as if
    # it were the model's genuine reply.
    if (
        result is not None
        and getattr(result, "is_error", False)
        and is_permanent_api_error_text(text)
    ):
        raise ClaudeSDKPermanentAPIError(
            f"Anthropic API rejected the request ({label}): {text}"
        )

    return text, result, reasoning

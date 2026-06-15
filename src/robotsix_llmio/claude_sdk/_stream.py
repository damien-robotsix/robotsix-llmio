"""Shared Claude Agent SDK streaming loop — used by both the no-tools
model path (``ClaudeSDKModel._invoke``) and the SDK-tools provider path
(``_SdkToolAgentHandle._invoke_query``).
"""

from __future__ import annotations

import asyncio
import json
import logging
from collections.abc import Callable
from typing import Any

log = logging.getLogger("robotsix_llmio.claude_sdk")


def _short(value: Any, limit: int = 200) -> str:
    """One-line, length-capped repr of a tool input / text for logging."""
    s = value if isinstance(value, str) else json.dumps(value, default=str)
    s = " ".join(s.split())
    return s if len(s) <= limit else s[:limit] + "…"


def _log_stream_message(message: Any, turn: list[int], label: str) -> None:
    """Emit a concise INFO line for one streamed Claude SDK message.

    Gives live feedback on what the agent is doing — turns, tool calls, tool
    results, the final result — even when Langfuse spans haven't flushed yet
    (a stuck agent never completes its span, so this is the only signal).
    *turn* is a 1-element list used as a mutable counter across the loop.
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
                elif bcls == "ToolUseBlock":
                    log.info(
                        "%s turn %d: tool_use %s(%s)",
                        label,
                        turn[0],
                        getattr(block, "name", "?"),
                        _short(getattr(block, "input", {})),
                    )
                elif bcls == "ThinkingBlock":
                    log.info(
                        "%s turn %d: thinking (%d chars)",
                        label,
                        turn[0],
                        len(getattr(block, "thinking", "") or ""),
                    )
        elif cls in ("UserMessage", "ToolResultMessage"):
            for block in getattr(message, "content", []) or []:
                if type(block).__name__ == "ToolResultBlock":
                    is_err = bool(getattr(block, "is_error", False))
                    log.info(
                        "%s tool_result%s — %s",
                        label,
                        " [ERROR]" if is_err else "",
                        _short(getattr(block, "content", "")),
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
        pass


async def _stream_query(
    prompt: str,
    options: Any,  # ClaudeAgentOptions (lazy import to avoid cycle with model)
    label: str,
    *,
    extra_transient: Callable[[Exception], bool] | None = None,
) -> tuple[str, Any]:
    """Run the Claude Agent SDK streaming loop under the per-call wall-clock cap.

    Returns ``(text, result)`` — joined assistant text (with the
    ``ResultMessage.result`` fallback) and the captured ``ResultMessage``.

    Converts a wall-clock timeout into :class:`ClaudeSDKQueryTimeout` (always).
    If *extra_transient* is given and returns ``True`` for a non-timeout
    exception, that exception is wrapped in :class:`ClaudeSDKTurnLimitError`.
    """
    from claude_agent_sdk import (
        AssistantMessage,
        ResultMessage,
        TextBlock,
        query,
    )

    from ..core import constants
    from .model import ClaudeSDKQueryTimeout, ClaudeSDKTurnLimitError

    chunks: list[str] = []
    result: Any = None
    turn = [0]

    async def _consume() -> None:
        nonlocal result
        async for message in query(prompt=prompt, options=options):
            _log_stream_message(message, turn, label)
            if isinstance(message, AssistantMessage):
                for block in message.content:
                    if isinstance(block, TextBlock):
                        chunks.append(block.text)
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
        if extra_transient is not None and extra_transient(exc):
            raise ClaudeSDKTurnLimitError(
                f"Claude Agent SDK hit the turn cap without "
                f"producing a final answer ({label}). The "
                f"agent loop did not converge — it kept taking turns instead "
                f"of terminating. This is a hard failure; retrying the "
                f"identical request would hit the cap again. SDK error: {exc}"
            ) from exc
        raise

    text = "".join(chunks).strip()
    if not text and result is not None:
        text = (getattr(result, "result", None) or "").strip()
    return text, result

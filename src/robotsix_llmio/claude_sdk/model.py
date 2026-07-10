"""Claude Agent SDK transport — a pydantic-ai ``Model`` over the ``claude`` CLI.

Drives the Claude Agent SDK (``claude_agent_sdk``) in **single-turn** mode and
adapts it to pydantic-ai's :class:`~pydantic_ai.models.Model` contract. The
appeal: it authenticates with your local ``claude login`` (Claude Code
subscription / OAuth) credentials — **no API key** — because the SDK spawns the
``claude`` CLI subprocess, which carries that auth.

Scope / limitations (by construction):
- The SDK runs its *own* agent loop and executes tools internally; it returns
  only final assistant text, never raw ``tool_use`` blocks. So this transport
  supports ``output_type=str`` and pydantic-ai's ``PromptedOutput`` (JSON in
  text), but **not** function/tool calling or the default tool-based structured
  output — those raise a clear :class:`UserError` instead of misbehaving.
- Every request spawns a fresh CLI subprocess and pays Claude Code's injected
  system-prompt overhead. This is a convenience transport, not a hot path.

Runtime requirements (beyond the ``claude_sdk`` extra): Node.js and the
``claude`` CLI installed and logged in (``claude login``).
"""

from __future__ import annotations

from typing import Any, cast

from pydantic_ai.exceptions import UserError
from pydantic_ai.messages import (
    ModelMessage,
    ModelRequest,
    ModelResponse,
    RetryPromptPart,
    SystemPromptPart,
    TextPart,
    ToolReturnPart,
    UserPromptPart,
)
from pydantic_ai.models import Model, ModelRequestParameters
from pydantic_ai.settings import ModelSettings
from pydantic_ai.usage import RequestUsage

from ..exceptions import RobotsixLLMIOError
from ._usage import _best_usage_dict, map_usage_dict
from .transient import is_claude_sdk_turn_limit

PROVIDER_NAME = "claude-sdk"

# Output modes this transport can satisfy with a plain text completion. The
# tool-based modes ('tool', 'native') need raw tool_use passthrough we can't do.
_TEXT_OUTPUT_MODES = {"text", "prompted"}

# Runaway backstop for the SDK agent loop — the single cap shared by BOTH
# transport paths:
#   * the no-tools Model path below (``allowed_tools=[]``) answers in a single
#     turn, so the cap is pure headroom; it must NOT be tight, because the SDK
#     *raises* ("Reached maximum number of turns") instead of returning the
#     answer if the budget is hit, so a low cap would false-trip on clean
#     answers;
#   * the injected-MCP-tools path (``provider._SdkToolAgentHandle``) runs a real
#     tool loop that legitimately needs many turns to converge.
# Hence a generous value: high enough that genuine tool loops don't trip it, low
# enough to stop a true runaway. If the cap IS reached, that is a HARD failure
# raised as ``ClaudeSDKTurnLimitError`` and never retried (retrying the identical
# request would just loop to the cap again) — fail loudly so the cause shows.
_MAX_TURNS = 100


class ClaudeSDKTurnLimitError(RobotsixLLMIOError):
    """The Claude Agent SDK loop hit its turn cap (``_MAX_TURNS``) without
    returning a final answer.

    A hard failure surfaced loudly: the agent loop did not converge, and the
    identical request would just loop to the cap again — so it is never treated
    as transient (see
    :func:`~robotsix_llmio.claude_sdk.transient.is_claude_sdk_transient`)."""


class ClaudeSDKQueryTimeout(RobotsixLLMIOError):
    """A single Claude Agent SDK ``query()`` exceeded the per-call wall-clock cap
    (:data:`~robotsix_llmio.core.constants.SDK_QUERY_TIMEOUT`).

    Unlike the turn-limit failure, a timeout is a *stall* (the subprocess made no
    progress — often startup contention), not a non-converging loop. Re-running
    usually clears it, so it is treated as **transient** and retried by the
    bounded retry (matched by name in
    :data:`~robotsix_llmio.claude_sdk.transient._SDK_TRANSIENT_NAMES`)."""


class ClaudeSDKUsageExhaustedError(RobotsixLLMIOError):
    """The Claude subscription has exhausted its usage credits for the
    ``ClaudeAgentOptions.model`` tier this call used.

    The SDK reports this as a normal-looking ``ResultMessage`` (``is_error=True``,
    often ``subtype="success"``) carrying the assistant-visible text "You're out
    of usage credits" rather than raising — so left unhandled, that text would be
    returned as if it were a genuine reply. A re-run at the *same* tier cannot
    help (the credits are exhausted until they reset), so this is never treated
    as transient (see
    :func:`~robotsix_llmio.claude_sdk.transient.is_claude_sdk_transient`) —
    callers should catch it and fall back to a different capability tier
    instead (e.g. via
    :func:`~robotsix_llmio.core.tier_fallback.acall_with_tier_fallback`)."""


def _binary_placeholder(item: Any) -> str:
    """Compact stand-in for a binary content part (image/audio/document).

    The SDK path is text-only, so binary payloads cannot reach the model —
    but they must NEVER be stringified either: ``str()`` on a pydantic-ai
    ``BinaryContent`` reprs the raw bytes (a 2 MB image becomes a ~6 MB
    escaped-byte string), which stalls the CLI subprocess for the full
    per-call wall-clock cap and hangs the caller.
    """
    media_type = getattr(item, "media_type", None) or "unknown type"
    size = len(getattr(item, "data", b"") or b"")
    return (
        f"[binary attachment: {media_type}, {size} bytes — "
        f"not visible to this text-only model]"
    )


def _is_binary_part(item: Any) -> bool:
    """True for content parts carrying a raw-bytes payload (``BinaryContent``)."""
    return isinstance(getattr(item, "data", None), (bytes, bytearray))


def _is_image_part(item: Any) -> bool:
    """True for binary parts whose media type is an image the SDK can display."""
    media_type = getattr(item, "media_type", "") or ""
    return _is_binary_part(item) and media_type.startswith("image/")


def extract_prompt_parts(user_prompt: Any) -> tuple[str, list[tuple[str, bytes]]]:
    """Split a caller prompt into ``(text, images)``.

    *user_prompt* may be a plain string, a bare content part, or a sequence of
    parts (pydantic-ai style: ``list[str | BinaryContent]``).  Image parts are
    returned as ``(media_type, raw_bytes)`` pairs for
    :func:`build_sdk_prompt`; non-image binary parts degrade to the
    :func:`_binary_placeholder` text.
    """
    if isinstance(user_prompt, str):
        return user_prompt, []
    if _is_image_part(user_prompt):
        return "", [(user_prompt.media_type, bytes(user_prompt.data))]
    if _is_binary_part(user_prompt):
        return _binary_placeholder(user_prompt), []
    if not isinstance(user_prompt, (list, tuple)):
        return str(user_prompt), []
    texts: list[str] = []
    images: list[tuple[str, bytes]] = []
    for item in user_prompt:
        if isinstance(item, str):
            texts.append(item)
        elif _is_image_part(item):
            images.append((item.media_type, bytes(item.data)))
        elif _is_binary_part(item):
            texts.append(_binary_placeholder(item))
        else:
            text = getattr(item, "text", None)
            texts.append(text if isinstance(text, str) else str(item))
    return "\n".join(texts), images


def build_sdk_prompt(
    text: str, images: list[tuple[str, bytes]]
) -> str | list[dict[str, Any]]:
    """Build the ``query()`` prompt: plain text, or streaming-input messages.

    With no images the text passes through unchanged (the SDK's simple string
    mode).  With images, returns a list of streaming-input message dicts — one
    user message whose content carries the text plus one base64 ``image``
    block per attachment, which the CLI accepts exactly like a pasted image.
    A **list** (not an async generator) so retries can safely re-send it;
    the stream layer wraps it in a fresh async iterator per attempt.
    """
    if not images:
        return text
    import base64

    content: list[dict[str, Any]] = []
    if text:
        content.append({"type": "text", "text": text})
    for media_type, data in images:
        content.append(
            {
                "type": "image",
                "source": {
                    "type": "base64",
                    "media_type": media_type,
                    "data": base64.b64encode(data).decode("ascii"),
                },
            }
        )
    return [{"type": "user", "message": {"role": "user", "content": content}}]


def collect_latest_user_images(
    messages: list[ModelMessage],
) -> list[tuple[str, bytes]]:
    """Image parts of the newest user turn, as ``(media_type, bytes)`` pairs.

    Only the latest ``UserPromptPart`` matters: earlier turns are replayed as
    text transcript (their images already degraded to placeholders when they
    were current).
    """
    for message in reversed(messages):
        if not isinstance(message, ModelRequest):
            continue
        for part in reversed(message.parts):
            if isinstance(part, UserPromptPart):
                content = part.content
                if isinstance(content, str):
                    return []
                if _is_image_part(content):
                    return [(content.media_type, bytes(content.data))]
                if isinstance(content, (list, tuple)):
                    return [
                        (item.media_type, bytes(item.data))
                        for item in content
                        if _is_image_part(item)
                    ]
                return []
    return []


def _content_to_text(content: Any) -> str:
    """Flatten a pydantic-ai user/tool content (str or a list of parts) to text.

    Binary parts are replaced with a short placeholder — see
    :func:`_binary_placeholder` for why they must not be stringified.
    """
    if isinstance(content, str):
        return content
    if _is_binary_part(content):
        return _binary_placeholder(content)
    if isinstance(content, (list, tuple)):
        out: list[str] = []
        for item in content:  # heterogeneous content parts
            text = getattr(item, "text", None)
            if isinstance(text, str):
                out.append(text)
            elif _is_binary_part(item):
                out.append(_binary_placeholder(item))
            else:
                out.append(str(item))
        return "\n".join(out)
    return str(content)


def _retry_text(part: RetryPromptPart) -> str:
    """The corrective text pydantic-ai wants shown back to the model on a retry
    (e.g. a JSON-validation failure during PromptedOutput)."""
    model_response = getattr(part, "model_response", None)
    if callable(model_response):
        try:
            return cast(str, model_response())
        except Exception:  # pragma: no cover - defensive
            pass
    return _content_to_text(getattr(part, "content", ""))


def render_prompt(messages: list[ModelMessage]) -> str:
    """Flatten the pydantic-ai message history into a single prompt string for
    the (stateless-per-call) SDK ``query``. A lone user turn is sent verbatim;
    multi-turn history is rendered as a labelled transcript so the model sees
    its own prior attempt and any correction."""
    turns: list[tuple[str, str]] = []
    for message in messages:
        if isinstance(message, ModelRequest):
            for part in message.parts:
                if isinstance(part, UserPromptPart):
                    turns.append(("user", _content_to_text(part.content)))
                elif isinstance(part, ToolReturnPart):
                    turns.append(
                        (
                            "user",
                            f"Tool result ({part.tool_name}): "
                            f"{_content_to_text(part.content)}",
                        )
                    )
                elif isinstance(part, RetryPromptPart):
                    turns.append(("user", _retry_text(part)))
        elif isinstance(message, ModelResponse):
            text = "\n".join(
                p.content for p in message.parts if isinstance(p, TextPart)
            )
            if text:
                turns.append(("assistant", text))

    if len(turns) == 1 and turns[0][0] == "user":
        return turns[0][1]
    return "\n\n".join(
        f"{'User' if role == 'user' else 'Assistant'}: {text}" for role, text in turns
    )


def _map_usage(result: Any) -> RequestUsage:
    """Map a Claude Agent SDK ``ResultMessage.usage`` (or ``model_usage``)
    dict onto pydantic-ai's :class:`RequestUsage`.

    Defensive: a missing/partial dict yields zeros.
    """
    usage = _best_usage_dict(result)
    return map_usage_dict(usage)


class ClaudeSDKModel(Model):
    """pydantic-ai model backed by the Claude Agent SDK (subscription auth).

    *sdk_model* is the value passed to the SDK's ``model`` option — a Claude
    Code alias (``"opus"``, ``"sonnet"``, ``"haiku"``) or a full model id.
    """

    def __init__(
        self,
        sdk_model: str,
        *,
        model_name: str | None = None,
        settings: ModelSettings | None = None,
    ) -> None:
        super().__init__(settings=settings)
        self._sdk_model = sdk_model
        self._model_name = model_name or sdk_model

    @property
    def model_name(self) -> str:
        """The model name reported to pydantic-ai (the SDK model id, or its
        override)."""
        return self._model_name

    @property
    def system(self) -> str:
        """The provider/system identifier for this model (always
        ``"anthropic"``)."""
        return "anthropic"

    @property
    def provider(self) -> None:
        """The HTTP provider for this model (always ``None``: the ``claude`` CLI
        subprocess is the transport, not an HTTP client)."""
        # No HTTP provider: the `claude` CLI subprocess is the transport, and
        # the SDK tears it down per call. The base ``__aenter__``/``__aexit__``
        # short-circuit on a None provider.
        return None

    # --- request ------------------------------------------------------------
    def _reject_unsupported(self, params: ModelRequestParameters) -> None:
        if params.function_tools:
            raise UserError(
                "ClaudeSDKModel does not support function/tool calling: the "
                "Claude Agent SDK executes tools inside its own loop and "
                "returns only final text. Build the agent without tools."
            )
        if params.output_mode not in _TEXT_OUTPUT_MODES:
            raise UserError(
                "ClaudeSDKModel supports only text or PromptedOutput results "
                f"(got output_mode={params.output_mode!r}). For structured "
                "output, wrap your type: output_type=PromptedOutput(MyModel)."
            )

    def _system_text(
        self, messages: list[ModelMessage], params: ModelRequestParameters
    ) -> str | None:
        """The system prompt for the SDK call: pydantic-ai's joined instructions
        (which already include any PromptedOutput JSON-schema directions) plus
        any classic ``SystemPromptPart`` content in the history."""
        parts: list[str] = []
        for message in messages:
            if isinstance(message, ModelRequest):
                parts.extend(
                    p.content
                    for p in message.parts
                    if isinstance(p, SystemPromptPart) and p.content
                )
        # pydantic-ai renamed/replaced the old ``_get_instructions`` (→ str)
        # with ``_get_instruction_parts`` (→ list[InstructionPart] | None);
        # join the parts' content into the system text.
        instruction_parts = self._get_instruction_parts(messages, params)
        if instruction_parts:
            parts.extend(p.content for p in instruction_parts if p.content)
        combined = "\n\n".join(dict.fromkeys(parts))  # de-dup, preserve order
        return combined or None

    async def _invoke(
        self, prompt: str | list[dict[str, Any]], system_text: str | None
    ) -> tuple[str, Any, str]:
        from claude_agent_sdk import (
            ClaudeAgentOptions,
        )

        from ._stream import _stream_query

        options = ClaudeAgentOptions(
            system_prompt=system_text,
            model=self._sdk_model,
            max_turns=_MAX_TURNS,  # backstop only; no tools => answers in one turn
            # This is the no-tools text path. ``allowed_tools=[]`` does NOT
            # disable the SDK's built-in tools (Bash/Read/Edit/Monitor/...) — an
            # empty allow-list means "no constraint", and ``can_use_tool`` is not
            # consulted for them. The reliable lever is ``disallowed_tools``; a
            # ``"*"`` wildcard denies every built-in tool (MCP tools, of which
            # there are none here, would be unaffected). ``bypassPermissions``
            # avoids a headless approval stall that otherwise degenerates into a
            # spurious "error result" when the model reaches for a denied tool.
            disallowed_tools=["*"],
            permission_mode="bypassPermissions",
            setting_sources=[],  # ignore project/user CLAUDE.md + settings
        )

        return await _stream_query(
            prompt,
            options,
            f"claude:{self._model_name}",
            extra_transient=is_claude_sdk_turn_limit,
        )

    async def request(
        self,
        messages: list[ModelMessage],
        model_settings: ModelSettings | None,
        model_request_parameters: ModelRequestParameters,
    ) -> ModelResponse:
        """Run *messages* through the Claude Agent SDK and return the final
        :class:`ModelResponse`.

        Rejects unsupported request parameters (function tools, non-text output
        modes), renders the conversation into a single prompt plus system text,
        and invokes the SDK to obtain the assistant's final text and usage.
        """
        self._reject_unsupported(model_request_parameters)
        system_text = self._system_text(messages, model_request_parameters)
        # Images from the newest user turn ride along as native SDK image
        # blocks (streaming-input mode); the transcript stays text.
        images = collect_latest_user_images(messages)
        prompt = build_sdk_prompt(render_prompt(messages), images)
        text, result, reasoning = await self._invoke(prompt, system_text)
        # Stamp the SDK's (estimated) cost onto the active span so the claude_sdk
        # provider logs cost in traces like the OpenRouter providers do.
        from ..core.cost import record_cost

        record_cost(
            result,
            lambda r: getattr(r, "total_cost_usd", None),
            provider=PROVIDER_NAME,
        )
        # Stamp provider/model identity and token usage on the active span,
        # independently of whether cost was recorded, so the span shape is
        # consistent with the tool-loop path and the OpenRouter provider.
        from ..core.tracing import (
            GEN_AI_OPERATION_NAME,
            GEN_AI_PROVIDER_NAME,
            GEN_AI_REQUEST_MODEL,
            GEN_AI_SYSTEM,
            GEN_AI_USAGE_INPUT_TOKENS,
            GEN_AI_USAGE_OUTPUT_TOKENS,
            LANGFUSE_OBSERVATION_METADATA_REASONING,
            OP_CHAT,
            get_recording_span,
        )

        span = get_recording_span()
        if span is not None:
            span.set_attribute(GEN_AI_OPERATION_NAME, OP_CHAT)
            span.set_attribute(GEN_AI_PROVIDER_NAME, PROVIDER_NAME)
            span.set_attribute(GEN_AI_SYSTEM, self.system)
            span.set_attribute(GEN_AI_REQUEST_MODEL, self._sdk_model)
            # Surface the model's extended-thinking content on the generation so
            # traces show the reasoning, not just the final answer + tool calls.
            if reasoning:
                span.set_attribute(LANGFUSE_OBSERVATION_METADATA_REASONING, reasoning)
            usage = _best_usage_dict(result)
            if usage is not None:
                in_tok = usage.get("input_tokens")
                out_tok = usage.get("output_tokens")
                if in_tok is not None:
                    span.set_attribute(GEN_AI_USAGE_INPUT_TOKENS, int(in_tok))
                if out_tok is not None:
                    span.set_attribute(GEN_AI_USAGE_OUTPUT_TOKENS, int(out_tok))
        return ModelResponse(
            parts=[TextPart(content=text)],
            usage=_map_usage(result),
            model_name=self._model_name,
            provider_name=PROVIDER_NAME,
            finish_reason="stop",
        )

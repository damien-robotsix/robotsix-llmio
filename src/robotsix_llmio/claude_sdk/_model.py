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

from typing import Any

from pydantic_ai.exceptions import UserError
from pydantic_ai.messages import (
    ModelMessage,
    ModelRequest,
    ModelResponse,
    SystemPromptPart,
    TextPart,
)
from pydantic_ai.models import Model, ModelRequestParameters
from pydantic_ai.settings import ModelSettings

from ._prompt import (
    _map_usage,
    build_sdk_prompt,
    collect_latest_user_images,
    render_prompt,
)
from ._task_budget import build_task_budget
from ._usage import _best_usage_dict
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
        max_tokens: int | None = None,
    ) -> None:
        super().__init__(settings=settings)
        self._sdk_model = sdk_model
        self._model_name = model_name or sdk_model
        self._max_tokens = max_tokens

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
            task_budget=build_task_budget(
                self._max_tokens, f"claude:{self._model_name}"
            ),
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

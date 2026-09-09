"""Direct-API DeepSeek model — reasoning round-trip + client-side cost.

A sibling of the OpenRouter transport model (``openrouter/_deepseek_model``)
that talks to ``https://api.deepseek.com`` directly. It reuses the DeepSeek-side
behaviours and drops the OpenRouter-only machinery:

- **Kept:** the thinking-mode ``reasoning_content`` round-trip — every assistant
  tool-call turn carries a ``reasoning_content`` STRING (the turn's real
  reasoning, else empty) so a reconstructed history (pre-seed, replayed
  ``conversation_state``, pause/resume mid tool-loop) is not rejected with HTTP
  400 ("The `reasoning_content` in the thinking mode must be passed back to the
  API."). Active only in thinking mode.
- **Dropped:** ``usage.include`` opt-in, ``usage.cost`` reads, and OpenRouter
  ``provider`` routing kwargs. The direct API returns token counts but no cost,
  so cost is computed client-side from configurable sticker prices (see
  :mod:`~robotsix_llmio.deepseek.pricing`).

DeepSeek selects thinking vs non-thinking by *model id* — ``deepseek-chat`` is
non-thinking and ``deepseek-reasoner`` is thinking — so the reasoning round-trip
is gated on the model id being the thinking model (set per level by the
provider), not on an OpenRouter ``reasoning`` request block.
"""

from __future__ import annotations

import logging
from typing import Any

from openai import AsyncStream
from openai.types.chat import ChatCompletionChunk
from pydantic_ai.models.openai import OpenAIChatModel

from .pricing import DeepseekPricing, record_deepseek_cost

logger = logging.getLogger(__name__)

#: DeepSeek's own API model ids (no OpenRouter ``deepseek/`` slug prefix).
NON_THINKING_MODEL = "deepseek-chat"
THINKING_MODEL = "deepseek-reasoner"

_REASONING_KEY = "reasoning"
_REASONING_CONTENT_KEY = "reasoning_content"
_TOOL_CALLS_KEY = "tool_calls"


def _reasoning_text(message: Any) -> str:
    """Concatenate the message's ``ThinkingPart`` contents into a string (the
    reasoning DeepSeek wants echoed back). Empty when the turn has no reasoning
    — e.g. a synthetic pre-seeded or reconstructed tool-call turn."""
    from pydantic_ai.messages import ThinkingPart

    parts = getattr(message, "parts", None) or []
    return "".join(
        p.content
        for p in parts
        if isinstance(p, ThinkingPart) and isinstance(getattr(p, "content", None), str)
    )


class DeepseekModel(OpenAIChatModel):
    """``OpenAIChatModel`` for the direct DeepSeek API.

    Carries the thinking-mode ``reasoning_content`` round-trip and records cost
    client-side (DeepSeek returns no ``usage.cost``). The provider stamps the
    per-level model id and :attr:`pricing` after construction.
    """

    #: Per-model sticker prices used to compute cost client-side. Stamped by the
    #: provider from its config; the default is used for direct construction.
    pricing: DeepseekPricing = DeepseekPricing()

    @property
    def _echo_reasoning(self) -> bool:
        """Carry ``reasoning_content`` on tool-call turns iff the model is the
        DeepSeek thinking model (``deepseek-reasoner``). The non-thinking model
        (``deepseek-chat``) needs no round-trip."""
        model_name = str(getattr(self, "model_name", "") or "")
        return model_name.startswith(THINKING_MODEL)

    async def _completions_create(self, *args: Any, **kwargs: Any) -> Any:
        # No usage.include opt-in and no provider-routing block on the direct
        # API — just record cost client-side once the response is available.
        response = await super()._completions_create(*args, **kwargs)
        if isinstance(response, AsyncStream):
            return _DeepseekCostCapturingStream(response, self.pricing)
        record_deepseek_cost(response, self.pricing)
        return response

    def _map_model_response(self, message: Any) -> Any:
        """Map a ModelResponse to an OpenAI assistant message, enforcing
        DeepSeek's thinking-mode reasoning rule (see module docstring).

        Thinking model: assistant tool-call turns carry ``reasoning_content`` (a
        string — the turn's real reasoning, else empty); non-tool-call turns and
        the non-thinking model carry no reasoning at all. The ``reasoning`` /
        ``reasoning_details`` variants are always dropped (DeepSeek rejects an
        array; only the string ``reasoning_content`` is accepted)."""
        param: Any = super()._map_model_response(message)
        if not (isinstance(param, dict) and param.get("role") == "assistant"):
            return param

        # Always clear the array/alias forms — DeepSeek only accepts the string.
        param.pop(_REASONING_KEY, None)
        param.pop("reasoning_details", None)

        if self._echo_reasoning and param.get(_TOOL_CALLS_KEY):
            # Present-but-possibly-empty string keeps the tool-call turn valid in
            # thinking mode even when the turn is synthetic/reconstructed.
            param[_REASONING_CONTENT_KEY] = _reasoning_text(message)
        else:
            param.pop(_REASONING_CONTENT_KEY, None)

        # DeepSeek rejects an assistant message with neither content nor
        # tool_calls (a thinking-only turn maps to no text and no tool calls). A
        # present empty string keeps such turns valid; this holds on every tier.
        if not param.get(_TOOL_CALLS_KEY) and not param.get("content"):
            param["content"] = ""
        return param


class _DeepseekCostCapturingStream:
    """Proxy around ``AsyncStream[ChatCompletionChunk]`` that records cost
    client-side from the final usage-bearing chunk on stream exhaustion.

    Satisfies the async-context-manager + async-iterator protocols that
    pydantic-ai requires of the ``_completions_create`` return value when
    ``stream=True``.
    """

    def __init__(
        self,
        stream: AsyncStream[ChatCompletionChunk],
        pricing: DeepseekPricing | None,
    ) -> None:
        self._stream = stream
        self._pricing = pricing
        self._last_usage_chunk: ChatCompletionChunk | None = None

    async def __aenter__(self) -> _DeepseekCostCapturingStream:
        await self._stream.__aenter__()
        return self

    async def __aexit__(self, *args: Any) -> None:
        await self._stream.__aexit__(*args)

    def __aiter__(self) -> _DeepseekCostCapturingStream:
        return self

    async def __anext__(self) -> ChatCompletionChunk:
        try:
            chunk: ChatCompletionChunk = await self._stream.__anext__()
            if chunk.usage is not None:
                self._last_usage_chunk = chunk
            return chunk
        except StopAsyncIteration:
            if self._last_usage_chunk is not None:
                record_deepseek_cost(self._last_usage_chunk, self._pricing)
            raise

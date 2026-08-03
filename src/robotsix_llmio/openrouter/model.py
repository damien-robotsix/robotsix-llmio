"""OpenRouter transport model — usage accounting + cost recording + prompt caching.

Model-family agnostic: this layer knows how to talk to OpenRouter (opt into
``usage.include`` and read ``usage.cost``) but applies no provider pin and no
reasoning policy. Those quirks live in derived modules (e.g.
``_deepseek_model``).

Prompt caching: the stable prefix of every request (system prompt + tool
schemas) is annotated with ``cache_control: {"type": "ephemeral"}`` markers so
that OpenRouter-compatible upstream providers (DeepSeek, Anthropic, …) cache it
and charge only cache-read rates (~10% of the full input price) on subsequent
turns instead of re-billing the full prefix as fresh input.

Why this matters: the mill worker loop resends the full system prompt and tool
schemas on every turn.  The input:output token ratio on the two highest-volume
models (deepseek-v4-pro and deepseek-v4-flash) runs ~176:1, so the prefix
dominates real cash spend.  Caching it moves those tokens from the "input"
bucket into the "cache read" bucket (roughly one-tenth the price).

Cache semantics confirmed (2026-08):
- OpenRouter normalizes per-content-block ``cache_control`` markers across
  providers.  The ``{"type": "ephemeral"}`` object on a message or tool
  definition tells the upstream to cache everything preceding it.
- DeepSeek honours these markers; the minimum cacheable prefix is 1,024 tokens
  and the cache TTL is ~5 minutes of active usage.
- Cache hits surface in ``usage.prompt_tokens_details.cached_tokens`` (already
  recorded on the OTel span by ``record_openrouter_cost``).
- Cache *writes* (the first request that populates the cache) are billed at a
  premium (~125% of the normal input price).  Subsequent reads are ~10%.
"""

from __future__ import annotations

import json
import logging
from typing import Any

from openai import AsyncStream
from openai.types import chat
from openai.types.chat import ChatCompletionChunk
from pydantic_ai.models.openai import OpenAIChatModel

from robotsix_llmio.core.tracing import (
    GEN_AI_OPERATION_NAME,
    GEN_AI_PROVIDER_NAME,
    GEN_AI_REQUEST_MODEL,
    GEN_AI_SYSTEM,
    GEN_AI_USAGE_CACHE_CREATION_INPUT_TOKENS,
    GEN_AI_USAGE_CACHE_READ_INPUT_TOKENS,
    GEN_AI_USAGE_COST,
    GEN_AI_USAGE_INPUT_TOKENS,
    GEN_AI_USAGE_OUTPUT_TOKENS,
    GEN_AI_USAGE_REASONING_TOKENS,
    LANGFUSE_COST_DETAILS_TOTAL_KEY,
    LANGFUSE_OBSERVATION_COST_DETAILS,
    LANGFUSE_OBSERVATION_METADATA_PROVIDER,
    OP_CHAT,
    get_recording_span,
)

logger = logging.getLogger(__name__)

PROVIDER_NAME: str = "openrouter"

#: ``cache_control`` marker for prompt caching on upstream providers that
#: support it (DeepSeek, Anthropic, etc.).  Placed on the last system message
#: and the last tool definition to mark the end of the cacheable prefix.
#: See module docstring for cache-semantics notes.
_CACHE_CONTROL_MARKER: dict[str, str] = {"type": "ephemeral"}


def _resolve_model_settings(args: tuple[Any, ...], kwargs: dict[str, Any]) -> Any:
    """Return the mutable ``model_settings`` dict from the parent call.
    Parent signature: ``(messages, stream, model_settings, params)``."""
    if "model_settings" in kwargs:
        return kwargs["model_settings"]
    if len(args) >= 3:
        return args[2]
    return None


def _inject_usage_include(args: tuple[Any, ...], kwargs: dict[str, Any]) -> None:
    """Merge ``extra_body.usage.include = True`` onto ``model_settings`` without
    trampling a caller-supplied ``extra_body``."""
    settings = _resolve_model_settings(args, kwargs)
    if settings is None:
        return
    extra_body = dict(settings.get("extra_body") or {})
    usage_opt = dict(extra_body.get("usage") or {})
    usage_opt.setdefault("include", True)
    extra_body["usage"] = usage_opt
    settings["extra_body"] = extra_body


def _get_cost_from_response(response: Any) -> float | None:
    """Extract the USD cost from an OpenRouter completion response, or ``None``
    when the response carries no usage/cost info."""
    usage_obj = getattr(response, "usage", None)
    if usage_obj is None:
        return None
    extras = getattr(usage_obj, "model_extra", None)
    raw_cost: Any = None
    if isinstance(extras, dict):
        raw_cost = extras.get("cost")
    if raw_cost is None:
        raw_cost = getattr(usage_obj, "cost", None)
    if raw_cost is None:
        return None
    try:
        return float(raw_cost)
    except TypeError, ValueError:
        return None


def record_openrouter_cost(response: Any) -> None:
    """Copy ``usage.cost`` (+ tokens + cache details + gen_ai attrs) onto the
    current OTel span. No-op without a cost, a recording span, or OpenTelemetry.

    Also emits an INFO-level log line summarising the cached vs. uncached input
    token split so the prompt-caching win is directly measurable in logs.
    """
    cost = _get_cost_from_response(response)
    if cost is None:
        return
    span = get_recording_span()
    if span is None:
        return

    usage_obj = getattr(response, "usage", None)
    span.set_attribute(GEN_AI_USAGE_COST, cost)
    span.set_attribute(
        LANGFUSE_OBSERVATION_COST_DETAILS,
        json.dumps({LANGFUSE_COST_DETAILS_TOTAL_KEY: cost}),
    )
    span.set_attribute(GEN_AI_OPERATION_NAME, OP_CHAT)
    span.set_attribute(GEN_AI_PROVIDER_NAME, PROVIDER_NAME)
    span.set_attribute(GEN_AI_SYSTEM, PROVIDER_NAME)
    # Provider tag Langfuse indexes onto the observation's metadata, so a
    # consumer can sum logged cost PER PROVIDER (cost reconciliation filters
    # the logged side to "openrouter" to match an OpenRouter key's scope).
    span.set_attribute(LANGFUSE_OBSERVATION_METADATA_PROVIDER, PROVIDER_NAME)

    model_name = getattr(response, "model", None)
    if model_name:
        span.set_attribute(GEN_AI_REQUEST_MODEL, model_name)

    cached_tokens: int | None = None
    cache_creation_tokens: int | None = None
    prompt_tokens: int | None = None

    if usage_obj is not None:
        prompt_tokens = getattr(usage_obj, "prompt_tokens", None)
        if prompt_tokens is not None:
            span.set_attribute(GEN_AI_USAGE_INPUT_TOKENS, prompt_tokens)
        completion_tokens = getattr(usage_obj, "completion_tokens", None)
        if completion_tokens is not None:
            span.set_attribute(GEN_AI_USAGE_OUTPUT_TOKENS, completion_tokens)
        prompt_details = getattr(usage_obj, "prompt_tokens_details", None)
        if prompt_details is not None:
            if isinstance(prompt_details, dict):
                cached = prompt_details.get("cached_tokens")
                cache_creation = prompt_details.get("cache_creation_input_tokens")
            else:
                cached = getattr(prompt_details, "cached_tokens", None)
                cache_creation = getattr(
                    prompt_details, "cache_creation_input_tokens", None
                )
            if cached is not None:
                cached_tokens = cached
                span.set_attribute(GEN_AI_USAGE_CACHE_READ_INPUT_TOKENS, cached)
            if cache_creation is not None:
                cache_creation_tokens = cache_creation
                span.set_attribute(
                    GEN_AI_USAGE_CACHE_CREATION_INPUT_TOKENS, cache_creation
                )
        completion_details = getattr(usage_obj, "completion_tokens_details", None)
        if completion_details is not None:
            if isinstance(completion_details, dict):
                reasoning = completion_details.get("reasoning_tokens")
            else:
                reasoning = getattr(completion_details, "reasoning_tokens", None)
            if reasoning is not None:
                span.set_attribute(GEN_AI_USAGE_REASONING_TOKENS, reasoning)

    # ------------------------------------------------------------------
    # Log cache-hit ratio so the win is measurable without a Langfuse UI.
    # ------------------------------------------------------------------
    if prompt_tokens is not None and cached_tokens is not None:
        cache_hit_pct = (cached_tokens / prompt_tokens * 100) if prompt_tokens else 0.0
        model_tag = f"{model_name} " if model_name else ""
        logger.info(
            "%sprompt tokens: total=%d cached=%d (%.1f%% hit)%s",
            model_tag,
            prompt_tokens,
            cached_tokens,
            cache_hit_pct,
            f" cache_creation={cache_creation_tokens}" if cache_creation_tokens else "",
        )


class _CostCapturingStream:
    """Proxy around ``AsyncStream[ChatCompletionChunk]`` that calls
    ``record_openrouter_cost`` with the final usage-bearing chunk on
    stream exhaustion.  Satisfies the async-context-manager + async-
    iterator protocols that pydantic-ai requires of the
    ``_completions_create`` return value when ``stream=True``.
    """

    def __init__(self, stream: AsyncStream[ChatCompletionChunk]) -> None:
        self._stream = stream
        self._last_usage_chunk: ChatCompletionChunk | None = None

    async def __aenter__(self) -> _CostCapturingStream:
        await self._stream.__aenter__()
        return self

    async def __aexit__(self, *args: Any) -> None:
        await self._stream.__aexit__(*args)

    def __aiter__(self) -> _CostCapturingStream:
        return self

    async def __anext__(self) -> ChatCompletionChunk:
        try:
            chunk: ChatCompletionChunk = await self._stream.__anext__()
            if chunk.usage is not None:
                self._last_usage_chunk = chunk
            return chunk
        except StopAsyncIteration:
            if self._last_usage_chunk is not None:
                record_openrouter_cost(self._last_usage_chunk)
            raise


class OpenRouterModel(OpenAIChatModel):
    """``OpenAIChatModel`` that opts into OpenRouter usage accounting, emits
    ``usage.cost`` onto the active OTel span, and annotates the stable request
    prefix with ``cache_control`` markers for prompt caching. No pin, no
    reasoning policy."""

    #: Whether to annotate the stable prefix (system messages + tools) with
    #: ``cache_control`` markers.  Enabled by default; subclasses or callers may
    #: set ``False`` to opt out (e.g. for providers that reject the marker).
    _prompt_caching_enabled: bool = True

    async def _map_messages(
        self, *args: Any, **kwargs: Any
    ) -> list[chat.ChatCompletionMessageParam]:
        messages = await super()._map_messages(*args, **kwargs)
        if not self._prompt_caching_enabled or not messages:
            return messages
        # Annotate the last system message so the system prompt is cached.
        for msg in reversed(messages):
            if msg.get("role") == "system":
                msg["cache_control"] = _CACHE_CONTROL_MARKER
                break
        return messages

    def _get_tool_choice(
        self, *args: Any, **kwargs: Any
    ) -> tuple[list[chat.ChatCompletionToolParam], Any]:
        tools, tool_choice = super()._get_tool_choice(*args, **kwargs)
        if self._prompt_caching_enabled and tools:
            # Annotate the last tool definition so the tool schemas are cached.
            tools[-1]["cache_control"] = _CACHE_CONTROL_MARKER
        return tools, tool_choice

    async def _completions_create(self, *args: Any, **kwargs: Any) -> Any:
        _inject_usage_include(args, kwargs)
        response = await super()._completions_create(*args, **kwargs)
        if isinstance(response, AsyncStream):
            return _CostCapturingStream(response)
        record_openrouter_cost(response)
        return response

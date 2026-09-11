"""Client-side cost for the direct DeepSeek API.

DeepSeek's OpenAI-compatible endpoint returns token counts but **no** cost (the
OpenRouter-only ``usage.cost`` field does not exist on the direct API), so cost
must be computed here from configurable per-1M-token sticker prices.

Prices are config-driven — a provider can be handed a ``pricing`` mapping via
its constructor (routed from a tier's ``provider_kwargs``), so the sticker
prices are updatable without a code change. The baked defaults below are
DeepSeek's published per-1M-token prices (source:
https://api-docs.deepseek.com/quick_start/pricing); override them via provider
config when a tariff changes.
"""

from __future__ import annotations

import logging
from collections.abc import Mapping
from typing import Any

from pydantic import BaseModel, Field

from ..core.cost import record_cost

logger = logging.getLogger(__name__)

#: Provider tag stamped onto the cost span so a consumer can sum logged cost
#: for the direct DeepSeek path independently of the OpenRouter slice.
PROVIDER_NAME: str = "deepseek"


class ModelPrices(BaseModel):
    """Per-1M-token USD prices for a single DeepSeek model id.

    Attributes:
        input: Price per 1M *uncached* (cache-miss) input tokens.
        output: Price per 1M output/completion tokens.
        cache_read: Price per 1M *cache-hit* input tokens (DeepSeek bills a
            cache read at a fraction of the full input rate).

    """

    input: float
    output: float
    cache_read: float


# Current sticker prices confirmed against DeepSeek's published pricing
# (https://api-docs.deepseek.com/quick_start/pricing): USD per 1M tokens.
# These are the published tariffs for these model ids at the time of writing;
# they are baked defaults, so a provider may override them via config
# (DeepseekPricing.from_mapping / provider_kwargs["pricing"]).
def _default_prices() -> dict[str, ModelPrices]:
    return {
        "deepseek-chat": ModelPrices(input=0.27, output=1.10, cache_read=0.07),
        "deepseek-reasoner": ModelPrices(input=0.55, output=2.19, cache_read=0.14),
    }


class DeepseekPricing(BaseModel):
    """Model-id → :class:`ModelPrices` map used to compute cost client-side.

    Populated from provider config so the sticker prices are updatable without
    a code change; unknown/unpriced model ids resolve to ``None`` and record no
    cost (see :func:`record_deepseek_cost`).
    """

    prices: dict[str, ModelPrices] = Field(default_factory=_default_prices)

    @classmethod
    def from_mapping(cls, mapping: Mapping[str, Any]) -> DeepseekPricing:
        """Build a pricing config by merging *mapping* over the baked defaults.

        *mapping* is a model-id → ``{"input", "output", "cache_read"}`` mapping
        (e.g. a tier's ``provider_kwargs["pricing"]``). Entries override the
        baked defaults for the same model id; unspecified models keep theirs.
        """
        merged = _default_prices()
        for model_id, value in mapping.items():
            merged[model_id] = (
                value if isinstance(value, ModelPrices) else ModelPrices(**value)
            )
        return cls(prices=merged)

    def for_model(self, model_id: str | None) -> ModelPrices | None:
        """Return the prices for *model_id*, or ``None`` when unpriced."""
        if not model_id:
            return None
        return self.prices.get(model_id)


def _usage_field(usage: Any, name: str) -> Any:
    """Read *name* from a usage object, falling back to its ``model_extra``.

    DeepSeek's cache-hit/miss counters (``prompt_cache_hit_tokens`` /
    ``prompt_cache_miss_tokens``) arrive as OpenAI-SDK *extra* fields, so they
    live in ``model_extra`` rather than as declared attributes.
    """
    val = getattr(usage, name, None)
    if val is None:
        extras = getattr(usage, "model_extra", None)
        if isinstance(extras, dict):
            val = extras.get(name)
    return val


def compute_cost(usage: Any, prices: ModelPrices | None) -> float | None:
    """Compute USD cost from a DeepSeek ``usage`` object and *prices*.

    Splits the input tokens into cache-hit (billed at ``cache_read``) and
    cache-miss (billed at ``input``) using DeepSeek's ``prompt_cache_hit_tokens``
    / ``prompt_cache_miss_tokens`` fields, falling back to
    ``prompt_tokens_details.cached_tokens`` and finally to treating every input
    token as uncached. Returns ``None`` when there is no usage or no prices.
    """
    if usage is None or prices is None:
        return None

    prompt_tokens = _usage_field(usage, "prompt_tokens") or 0
    completion_tokens = _usage_field(usage, "completion_tokens") or 0

    cache_hit = _usage_field(usage, "prompt_cache_hit_tokens")
    cache_miss = _usage_field(usage, "prompt_cache_miss_tokens")
    if cache_hit is None and cache_miss is None:
        details = _usage_field(usage, "prompt_tokens_details")
        if isinstance(details, dict):
            cache_hit = details.get("cached_tokens")
        elif details is not None:
            cache_hit = getattr(details, "cached_tokens", None)

    cached = cache_hit or 0
    uncached = cache_miss if cache_miss is not None else max(prompt_tokens - cached, 0)

    return (
        uncached * prices.input
        + cached * prices.cache_read
        + completion_tokens * prices.output
    ) / 1_000_000


def record_deepseek_cost(response: Any, pricing: DeepseekPricing | None) -> None:
    """Compute client-side cost for *response* and record it via
    :func:`~robotsix_llmio.core.cost.record_cost` with ``provider="deepseek"``.

    Looks up per-1M prices for the response's model id; an unknown/unpriced
    model id records nothing (debug log) and never raises.
    """
    model_id = getattr(response, "model", None)
    prices = pricing.for_model(model_id) if pricing is not None else None
    if prices is None:
        logger.debug("No DeepSeek pricing for model %r; recording no cost.", model_id)
        return
    record_cost(
        response,
        lambda r: compute_cost(getattr(r, "usage", None), prices),
        provider=PROVIDER_NAME,
    )

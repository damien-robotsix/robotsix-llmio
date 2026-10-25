"""Client-side cost for the direct Anthropic Messages API.

The Messages API returns token counts — uncached input, output, cache-read and
cache-write (``cache_creation_input_tokens``) — but **no** dollar cost, so cost
is computed here from configurable per-1M-token sticker prices.

Prices are config-driven — a provider can be handed a ``pricing`` mapping via
its constructor (routed from a tier's ``provider_kwargs``), so the sticker
prices are updatable without a code change. The baked defaults below are
Anthropic's published first-party per-1M-token prices (source:
https://platform.claude.com/docs/en/about-claude/pricing); cache writes are
priced at the 5-minute TTL rate (1.25x input). Override them via provider
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
#: for the direct Anthropic API path independently of the Claude SDK slice.
PROVIDER_NAME: str = "anthropic"


class ModelPrices(BaseModel):
    """Per-1M-token USD prices for a single Anthropic model id.

    Attributes:
        input: Price per 1M *uncached* input tokens.
        output: Price per 1M output tokens (thinking tokens bill as output).
        cache_read: Price per 1M cache-hit input tokens.
        cache_write: Price per 1M cache-write input tokens (5-minute TTL rate).

    """

    input: float
    output: float
    cache_read: float
    cache_write: float


def _prices(inp: float, out: float, cache_read: float | None = None) -> ModelPrices:
    """Build prices with Anthropic's standard multipliers: cache read 0.1x
    input (unless given explicitly), 5-minute cache write 1.25x input."""
    return ModelPrices(
        input=inp,
        output=out,
        cache_read=inp * 0.1 if cache_read is None else cache_read,
        cache_write=inp * 1.25,
    )


# Current first-party sticker prices (USD per 1M tokens) for the Claude models
# in use. Baked defaults — override via AnthropicPricing.from_mapping /
# provider_kwargs["pricing"] when a tariff changes.
def _default_prices() -> dict[str, ModelPrices]:
    return {
        "claude-fable-5-1": _prices(10.0, 50.0, cache_read=0.25),
        "claude-fable-5": _prices(10.0, 50.0),
        "claude-opus-5-5": _prices(4.0, 20.0, cache_read=0.20),
        "claude-opus-5": _prices(5.0, 25.0),
        "claude-opus-4-8": _prices(5.0, 25.0),
        "claude-opus-4-7": _prices(5.0, 25.0),
        "claude-sonnet-5-5": _prices(2.0, 10.0),
        "claude-sonnet-5": _prices(2.0, 10.0),
        "claude-sonnet-4-6": _prices(3.0, 15.0),
        "claude-haiku-5-5": _prices(0.10, 0.50),
        "claude-haiku-4-5": _prices(1.0, 5.0),
    }


class AnthropicPricing(BaseModel):
    """Model-id → :class:`ModelPrices` map used to compute cost client-side.

    Unknown/unpriced model ids resolve to ``None`` and record no cost (see
    :func:`record_anthropic_cost`).
    """

    prices: dict[str, ModelPrices] = Field(default_factory=_default_prices)

    @classmethod
    def from_mapping(cls, mapping: Mapping[str, Any]) -> AnthropicPricing:
        """Build a pricing config by merging *mapping* over the baked defaults.

        *mapping* is a model-id → ``{"input", "output", "cache_read",
        "cache_write"}`` mapping (e.g. a tier's ``provider_kwargs["pricing"]``).
        Entries override the baked defaults for the same model id.
        """
        merged = _default_prices()
        for model_id, value in mapping.items():
            merged[model_id] = (
                value if isinstance(value, ModelPrices) else ModelPrices(**value)
            )
        return cls(prices=merged)

    def for_model(self, model_id: str | None) -> ModelPrices | None:
        """Return the prices for *model_id*, or ``None`` when unpriced.

        A dated snapshot id (e.g. ``claude-haiku-4-5-20251001``) falls back to
        the longest priced alias it starts with.
        """
        if not model_id:
            return None
        if model_id in self.prices:
            return self.prices[model_id]
        candidates = [k for k in self.prices if model_id.startswith(k + "-")]
        if not candidates:
            return None
        return self.prices[max(candidates, key=len)]


def compute_cost(usage: Any, prices: ModelPrices | None) -> float | None:
    """Compute USD cost from a pydantic-ai ``RequestUsage`` and *prices*.

    pydantic-ai's ``input_tokens`` for Anthropic is the TOTAL input (uncached +
    cache read + cache write), with the cache split in ``cache_read_tokens`` /
    ``cache_write_tokens``; the uncached part is the remainder. Returns
    ``None`` when there is no usage or no prices.
    """
    if usage is None or prices is None:
        return None
    input_tokens = getattr(usage, "input_tokens", 0) or 0
    output_tokens = getattr(usage, "output_tokens", 0) or 0
    cache_read = getattr(usage, "cache_read_tokens", 0) or 0
    cache_write = getattr(usage, "cache_write_tokens", 0) or 0
    uncached = max(input_tokens - cache_read - cache_write, 0)
    return (
        uncached * prices.input
        + cache_read * prices.cache_read
        + cache_write * prices.cache_write
        + output_tokens * prices.output
    ) / 1_000_000


def record_anthropic_cost(
    usage: Any, model_id: str | None, pricing: AnthropicPricing | None
) -> None:
    """Compute client-side cost for *usage* and record it via
    :func:`~robotsix_llmio.core.cost.record_cost` with ``provider="anthropic"``.

    An unknown/unpriced *model_id* records nothing (debug log) and never raises.
    """
    prices = pricing.for_model(model_id) if pricing is not None else None
    if prices is None:
        logger.debug("No Anthropic pricing for model %r; recording no cost.", model_id)
        return
    record_cost(usage, lambda u: compute_cost(u, prices), provider=PROVIDER_NAME)

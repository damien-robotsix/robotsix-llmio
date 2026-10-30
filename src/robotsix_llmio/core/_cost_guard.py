"""Per-request cost guard — pure helpers for failover cost estimation.

Extracted from :mod:`robotsix_llmio.core.failover`: stateless arithmetic
estimating a slot's worst-case per-request cost and deciding whether a
failover slot switch is too expensive. The failover loop consults
:func:`_cost_guard_verdict` before switching slots.
"""

from __future__ import annotations

from robotsix_llmio.config.tier import FailoverConfig, TierLevelConfig

#: Scale factor turning a per-1M-token price into a per-token price.
_TOKENS_PER_PRICE_UNIT: float = 1_000_000.0


def estimate_request_cost(tlc: TierLevelConfig) -> float | None:
    """Best-effort worst-case USD cost of one request on *tlc*.

    Derived from the slot's configured price ceiling and output cap: the
    ``max_price_completion`` provider kwarg (USD per 1M completion tokens)
    times the ``max_tokens`` output cap. This is the knowable worst-case
    spend for a single request on that slot.

    Returns ``None`` when the slot carries no completion price ceiling or no
    output cap — the Claude SDK ``default`` slots, for instance, expose
    neither — so callers can treat the estimate as *unavailable* rather than
    as a free (zero-cost) slot.
    """
    max_tokens = tlc.max_tokens
    price_completion = tlc.provider_kwargs.get("max_price_completion")
    if max_tokens is None or price_completion is None:
        return None
    try:
        return float(price_completion) * max_tokens / _TOKENS_PER_PRICE_UNIT
    except TypeError, ValueError:
        return None


def _cost_guard_verdict(
    primary: TierLevelConfig,
    candidate: TierLevelConfig,
    config: FailoverConfig,
) -> str | None:
    """Return a refusal reason when failover to *candidate* is too expensive.

    Two independent, config-driven checks run against the estimated
    worst-case per-request cost of *candidate* (see
    :func:`estimate_request_cost`):

    * the per-request **ceiling** ``max_request_cost_usd`` — refuse when the
      candidate estimate *strictly* exceeds it;
    * the **ratio** ``cost_ratio_threshold`` — refuse when the candidate
      estimate *strictly* exceeds the *primary* estimate times the threshold
      (the candidate is materially more expensive than staying put).

    Returns ``None`` (failover allowed) when neither check is configured,
    when the candidate cost cannot be estimated (nothing to price), or when
    the candidate is same-or-cheaper. A missing *primary* estimate disables
    only the ratio check; the ceiling check still applies. The boundary is
    inclusive — an estimate exactly equal to the ceiling (or exactly the
    primary times the ratio) is allowed.
    """
    ceiling = config.max_request_cost_usd
    ratio = config.cost_ratio_threshold
    if ceiling is None and ratio is None:
        return None

    candidate_cost = estimate_request_cost(candidate)
    if candidate_cost is None:
        return None

    if ceiling is not None and candidate_cost > ceiling:
        return (
            f"estimated fallback cost ${candidate_cost:.4f} exceeds the "
            f"per-request ceiling ${ceiling:.4f}"
        )

    if ratio is not None:
        primary_cost = estimate_request_cost(primary)
        if primary_cost is not None and candidate_cost > primary_cost * ratio:
            return (
                f"estimated fallback cost ${candidate_cost:.4f} is more than "
                f"{ratio:g}x the primary slot's ${primary_cost:.4f}"
            )

    return None

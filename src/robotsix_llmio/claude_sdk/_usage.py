"""Shared Claude Agent SDK usage-dict → pydantic-ai ``RequestUsage`` mapping."""

from __future__ import annotations

from typing import Any

from pydantic_ai.usage import RequestUsage

# CamelCase keys found in the per-model ``model_usage`` sub-dicts →
# canonical snake_case keys used downstream.
_CAMEL_TO_SNAKE: dict[str, str] = {
    "inputTokens": "input_tokens",
    "outputTokens": "output_tokens",
    "cacheReadInputTokens": "cache_read_input_tokens",
    "cacheCreationInputTokens": "cache_creation_input_tokens",
}


def _aggregate_per_model(d: dict[str, Any]) -> dict[str, Any] | None:
    """Aggregate a per-model ``model_usage`` dict into a flat snake_case dict.

    The current SDK wire shape is::

        {"claude-3-5-haiku-20241022": {"inputTokens": 200, "outputTokens": 50, …}}

    Returns ``None`` when *d* is not in the per-model format (its values
    are not dicts, or no recognised token keys are present).
    """
    total: dict[str, int] = {}
    for sub in d.values():
        if not isinstance(sub, dict):
            return None
        for camel, snake in _CAMEL_TO_SNAKE.items():
            val = sub.get(camel)
            if val is not None:
                total[snake] = total.get(snake, 0) + int(val)
    return total or None


def _best_usage_dict(result: Any) -> dict[str, Any] | None:
    """Return the most-promising token-usage dict from an SDK *result*.

    The Claude Agent SDK ``ResultMessage`` carries two parallel usage fields:
    ``model_usage`` (the per-model consumption the platform tracks) and
    ``usage`` (legacy aggregate).  Prefer *model_usage* when it can be
    normalised to the canonical ``input_tokens`` / ``output_tokens`` keys;
    fall back to *usage*.
    """
    if result is None:
        return None
    for attr in ("model_usage", "usage"):
        d = getattr(result, attr, None)
        if not isinstance(d, dict) or not d:
            continue
        # Flat snake_case format (legacy ``usage``, or test fixtures):
        if "input_tokens" in d and "output_tokens" in d:
            return d
        # Per-model camelCase format (current SDK ``model_usage`` wire shape):
        aggregated = _aggregate_per_model(d)
        if aggregated is not None:
            return aggregated
        # Non-empty dict without canonical keys — best-effort fallback
        # (may carry tokens under alternate key names, e.g. from an older SDK).
        if d:
            return d
    return None


def map_usage_dict(usage: Any) -> RequestUsage:
    """Map a Claude Agent SDK usage dict onto pydantic-ai's RequestUsage.

    Defensive: a missing/non-dict/partial value yields zeros.
    """
    if not isinstance(usage, dict):
        return RequestUsage()
    return RequestUsage(
        input_tokens=int(usage.get("input_tokens") or 0),
        output_tokens=int(usage.get("output_tokens") or 0),
        cache_read_tokens=int(usage.get("cache_read_input_tokens") or 0),
        cache_write_tokens=int(usage.get("cache_creation_input_tokens") or 0),
    )

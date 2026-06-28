"""Shared Claude Agent SDK usage-dict → pydantic-ai ``RequestUsage`` mapping."""

from __future__ import annotations

from typing import Any

from pydantic_ai.usage import RequestUsage


def _best_usage_dict(result: Any) -> dict[str, Any] | None:
    """Return the most-promising token-usage dict from an SDK *result*.

    The Claude Agent SDK ``ResultMessage`` carries two parallel usage fields:
    ``model_usage`` (the per-model consumption the platform tracks) and
    ``usage`` (legacy aggregate).  Prefer *model_usage* when it contains the
    standard ``input_tokens`` / ``output_tokens`` keys; fall back to *usage*.
    """
    if result is None:
        return None
    for attr in ("model_usage", "usage"):
        d = getattr(result, attr, None)
        if isinstance(d, dict) and "input_tokens" in d and "output_tokens" in d:
            return d
        if isinstance(d, dict) and d:
            # A non-empty dict without the canonical keys may still carry
            # tokens under alternate key names (e.g. from an older SDK).
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

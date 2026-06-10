"""Shared Claude Agent SDK usage-dict → pydantic-ai ``RequestUsage`` mapping."""

from __future__ import annotations

from typing import Any

from pydantic_ai.usage import RequestUsage


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

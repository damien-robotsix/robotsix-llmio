"""Tier-escalation loop for four-tier provider+model fallback.

This module provides a dispatch helper that iterates through the four
configured tiers in :class:`~robotsix_llmio.config.tier.TierConfig`
automatically, catching failures at one level and promoting to the next.

It does **not** perform local retries — those remain the responsibility
of the callable produced by *fn_factory* (which typically wraps
:func:`~robotsix_llmio.core.retry.call_with_retry` with provider-specific
transient detection).

Fallback direction
------------------

1. On failure at the current tier, the next **higher** unvisited tier is tried.
2. If no higher unvisited tier exists, the next **lower** unvisited tier is tried.
3. A tier is never revisited — the visited-set prevents ping-pong cycles.

========  ==================================================
Start     Chain on successive failures (default depth 2)
========  ==================================================
LEVEL1    LEVEL1 → LEVEL2 → LEVEL3 → stop
LEVEL2    LEVEL2 → LEVEL3 → LEVEL4 → stop
LEVEL3    LEVEL3 → LEVEL4 → LEVEL2 → stop
LEVEL4    LEVEL4 → LEVEL3 → LEVEL2 → stop
========  ==================================================

(The default ``max_fallback_depth=2`` allows two promotions, so at most
three of the four tiers are visited per call.)
"""

from __future__ import annotations

import asyncio
import logging
import time
from collections.abc import Awaitable, Callable
from typing import Any, TypeVar

from robotsix_llmio.config.tier import TierConfig, TierLevel, TierLevelConfig

log = logging.getLogger("robotsix_llmio.tier_fallback")

T = TypeVar("T")

# Ordered tuple of all TierLevel members, used by _next_unvisited_tier
# to determine priority by position (LEVEL1=0, LEVEL2=1, LEVEL3=2, LEVEL4=3).
_ALL_TIER_LEVELS: tuple[TierLevel, ...] = tuple(TierLevel)


def _next_unvisited_tier(
    current: TierLevel,
    visited: frozenset[TierLevel],
) -> TierLevel | None:
    """Return the next unvisited tier preferring higher, then lower.

    Candidates are considered in priority order:

    1. Higher tiers: iterate ``_ALL_TIER_LEVELS`` in ordinal order,
       select the first where ordinal > *current* and the tier is not
       in *visited*.
    2. Lower tiers: iterate ``_ALL_TIER_LEVELS`` in **reverse** ordinal
       order (nearest-first), select the first where ordinal < *current*
       and the tier is not in *visited*.

    Returns ``None`` when no unvisited tier remains.
    """
    # Find current position
    try:
        current_idx = _ALL_TIER_LEVELS.index(current)
    except ValueError:
        return None

    # 1. Higher tiers: scan forward from current_idx+1
    for idx in range(current_idx + 1, len(_ALL_TIER_LEVELS)):
        candidate = _ALL_TIER_LEVELS[idx]
        if candidate not in visited:
            return candidate

    # 2. Lower tiers: scan backward from current_idx-1 (nearest-first)
    for idx in range(current_idx - 1, -1, -1):
        candidate = _ALL_TIER_LEVELS[idx]
        if candidate not in visited:
            return candidate

    return None


async def _tier_fallback_loop(
    fn_factory: Callable[[TierLevelConfig], Callable[[], Any]],
    *,
    invoke: Callable[[Callable[[], Any]], Awaitable[Any]],
    tier_config: TierConfig,
    level: TierLevel = TierLevel.LEVEL1,
    fallback_enabled: bool = False,
    max_fallback_depth: int = 2,
    what: str = "model call",
) -> Any:
    """Shared tier-escalation loop — ``invoke`` adapts sync vs async."""
    if level not in _ALL_TIER_LEVELS:
        raise ValueError(f"Unknown tier level: {level!r}")

    visited: set[TierLevel] = {level}
    promotions = 0
    current_level = level

    while True:
        tlc: TierLevelConfig = getattr(tier_config, current_level.value)

        log.info(
            "%s: trying %s (provider=%s, model=%s)",
            what,
            current_level.value,
            tlc.provider,
            tlc.model_name,
        )

        try:
            callable_fn = fn_factory(tlc)
            result = await invoke(callable_fn)
        except Exception as exc:
            next_level = _next_unvisited_tier(current_level, frozenset(visited))

            if next_level is None:
                raise

            exhausted = not fallback_enabled or promotions >= max_fallback_depth
            if exhausted:
                raise

            log.warning(
                "%s: %s failed with %s — falling back to %s",
                what,
                current_level.value,
                type(exc).__name__,
                next_level.value,
            )

            visited.add(next_level)
            promotions += 1
            current_level = next_level
            continue

        log.info("%s: %s succeeded", what, current_level.value)
        return result


def call_with_tier_fallback[T](
    fn_factory: Callable[[TierLevelConfig], Callable[[], T]],
    *,
    tier_config: TierConfig,
    level: TierLevel = TierLevel.LEVEL1,
    fallback_enabled: bool = False,
    max_fallback_depth: int = 2,
    what: str = "model call",
    sleep: Callable[[float], None] = time.sleep,
) -> T:
    """Run *fn_factory*-produced callables with tier escalation.

    Parameters
    ----------
    fn_factory:
        Called with the :class:`TierLevelConfig` for the current tier level;
        must return a zero-argument callable that performs the actual work
        (typically a model call with provider-specific retry). Called
        **fresh** for each tier level visited.
    tier_config:
        A validated :class:`TierConfig` providing ``level1``, ``level2``,
        ``level3``, ``level4`` :class:`TierLevelConfig` attributes.
    level:
        The starting :class:`TierLevel` (default ``LEVEL1``).
    fallback_enabled:
        When ``False`` (default), only the starting level is tried; any
        failure re-raises immediately. When ``True``, escalation proceeds
        according to *max_fallback_depth*.
    max_fallback_depth:
        Maximum number of tier promotions (tier switches) allowed. Default
        ``2`` means at most two promotions, so up to three tiers can be
        tried. ``0`` means no escalation.
    what:
        Human-readable label for log messages (default ``"model call"``).
    sleep:
        Injectable sleep for tests (default :func:`time.sleep`; reserved
        for future backoff between tier promotions).
    """

    async def _invoke(f: Callable[[], Any]) -> Any:
        return f()

    return asyncio.run(  # type: ignore[no-any-return]
        _tier_fallback_loop(
            fn_factory,
            invoke=_invoke,
            tier_config=tier_config,
            level=level,
            fallback_enabled=fallback_enabled,
            max_fallback_depth=max_fallback_depth,
            what=what,
        )
    )


async def acall_with_tier_fallback[T](
    fn_factory: Callable[[TierLevelConfig], Callable[[], Awaitable[T]]],
    *,
    tier_config: TierConfig,
    level: TierLevel = TierLevel.LEVEL1,
    fallback_enabled: bool = False,
    max_fallback_depth: int = 2,
    what: str = "model call",
    sleep: Callable[[float], Awaitable[None]] = asyncio.sleep,
) -> T:
    """Async mirror of :func:`call_with_tier_fallback`.

    Identical semantics except *fn_factory* returns an awaitable-producing
    callable, the callable is awaited, and *sleep* defaults to
    :func:`asyncio.sleep`.
    """

    async def _invoke(f: Callable[[], Any]) -> Any:
        return await f()

    return await _tier_fallback_loop(  # type: ignore[no-any-return]
        fn_factory,
        invoke=_invoke,
        tier_config=tier_config,
        level=level,
        fallback_enabled=fallback_enabled,
        max_fallback_depth=max_fallback_depth,
        what=what,
    )

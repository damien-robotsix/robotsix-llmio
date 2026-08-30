"""Tier-escalation loop for five-tier provider+model fallback.

This module provides a dispatch helper that iterates through the five
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
three of the five tiers are visited per call.)

Fallback is **on by default** (``fallback_enabled=True``). A tier binding can
fail for reasons that have nothing to do with the request — a provider outage,
or a subscription whose usage credits are exhausted until they reset — and in
those cases the work is better done by another tier than not done at all.

Two tiers may share one backend (the baked defaults put several tiers on the
Claude SDK subscription). When a level fails with a **provider-wide
exhaustion** — a :class:`~robotsix_llmio.exceptions.ProviderExhaustedError`
such as ``ClaudeSDKUsageExhaustedError`` — every other level on that same
provider is also spent, so the loop marks them all visited and skips past them
in a single step (regardless of ``max_fallback_depth``) rather than wasting
fallback hops walking sibling tiers that would only fail the same way. Ordinary
failures (transient errors, per-run rate limits) still fall back one level at a
time. Pass ``fallback_enabled=False`` where a caller depends on one specific
tier's behaviour rather than on getting an answer.
"""

from __future__ import annotations

import asyncio
import logging
import time
from collections.abc import Awaitable, Callable
from typing import Any, TypeVar

from robotsix_llmio.config.tier import TierConfig, TierLevel, TierLevelConfig
from robotsix_llmio.exceptions import ProviderExhaustedError

from ._otel import get_recording_span, get_tracer, start_span
from .cooldown import get_health_tracker
from .retry import _drive_sync, _walk_cause_chain

log = logging.getLogger("robotsix_llmio.tier_fallback")

_TRACER_NAME: str = "robotsix_llmio.core.tier_fallback"

# OTel span attribute keys for tier-fallback observability.
# Centralised constants so a rename is a one-line change.
_ATTR_TIER_LEVEL = "llmio.tier.level"
_ATTR_TIER_PROVIDER = "llmio.tier.provider"
_ATTR_TIER_MODEL = "llmio.tier.model"
_ATTR_TIER_ATTEMPT_INDEX = "llmio.tier.attempt_index"
_ATTR_TIER_SUCCEEDED = "llmio.tier.succeeded"
_ATTR_TIER_PROMOTIONS = "llmio.tier.promotions"
_ATTR_TIER_FALLBACK_ACTIVATED = "llmio.tier.fallback_activated"
_ATTR_TIER_COOLDOWN_SKIPPED = "llmio.tier.cooldown_skipped"
_ATTR_TIER_COOLDOWN_REASON = "llmio.tier.cooldown_reason"

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


def _is_provider_exhausted(exc: BaseException) -> bool:
    """True when *exc* signals a provider-wide exhaustion.

    Matches :class:`~robotsix_llmio.exceptions.ProviderExhaustedError` (e.g.
    ``ClaudeSDKUsageExhaustedError``) anywhere in the cause/context chain — a
    backend out of capacity until a quota resets, shared by every tier level on
    that provider. Deliberately excludes per-run rate-limits
    (``UsageLimitExceeded``), which are not provider-wide and still fall back
    level-by-level.
    """
    return any(
        isinstance(cur, ProviderExhaustedError) for cur in _walk_cause_chain(exc)
    )


def _levels_on_provider(tier_config: TierConfig, provider: str) -> set[TierLevel]:
    """Return every :class:`TierLevel` in *tier_config* backed by *provider*."""
    return {
        lvl
        for lvl in _ALL_TIER_LEVELS
        if getattr(tier_config, lvl.value).provider == provider
    }


async def _tier_fallback_loop(
    fn_factory: Callable[[TierLevelConfig], Callable[[], Any]],
    *,
    invoke: Callable[[Callable[[], Any]], Awaitable[Any]],
    tier_config: TierConfig,
    level: TierLevel = TierLevel.LEVEL1,
    fallback_enabled: bool = True,
    max_fallback_depth: int = 2,
    what: str = "model call",
) -> Any:
    """Shared tier-escalation loop — ``invoke`` adapts sync vs async."""
    if level not in _ALL_TIER_LEVELS:
        raise ValueError(f"Unknown tier level: {level!r}")

    visited: set[TierLevel] = {level}
    promotions = 0
    current_level = level

    run_span = get_recording_span()

    health_tracker = get_health_tracker()

    while True:
        tlc: TierLevelConfig = getattr(tier_config, current_level.value)

        # ---- cooldown check: skip models that are currently unhealthy ----
        if health_tracker.is_in_cooldown(tlc.model):
            log.info(
                "%s: %s (model=%s) is in cooldown — skipping",
                what,
                current_level.value,
                tlc.model,
            )
            if run_span is not None:
                run_span.set_attribute(_ATTR_TIER_COOLDOWN_SKIPPED, True)
                run_span.set_attribute(_ATTR_TIER_COOLDOWN_REASON, current_level.value)

            next_level = _next_unvisited_tier(current_level, frozenset(visited))

            if next_level is None:
                raise RuntimeError(
                    f"{what}: all tiers exhausted; "
                    f"{current_level.value} is in cooldown and no unvisited "
                    f"tier remains"
                )

            exhausted = not fallback_enabled or promotions >= max_fallback_depth
            if exhausted:
                raise RuntimeError(
                    f"{what}: {current_level.value} is in cooldown and "
                    f"fallback depth ({max_fallback_depth}) exhausted"
                )

            if run_span is not None:
                run_span.set_attribute(_ATTR_TIER_PROMOTIONS, promotions + 1)
                run_span.set_attribute(_ATTR_TIER_FALLBACK_ACTIVATED, True)

            visited.add(next_level)
            promotions += 1
            current_level = next_level
            continue

        # ---- attempt the tier ----
        with start_span(
            get_tracer(_TRACER_NAME),
            "llmio.tier.attempt",
            attributes={
                _ATTR_TIER_LEVEL: current_level.value,
                _ATTR_TIER_PROVIDER: tlc.provider,
                _ATTR_TIER_MODEL: tlc.model_name,
                _ATTR_TIER_ATTEMPT_INDEX: len(visited),
            },
        ) as span:
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
                if span is not None:
                    span.set_attribute(_ATTR_TIER_SUCCEEDED, False)

                # Record terminal failures for cooldown tracking.
                health_tracker.record_failure(tlc.model, exc=exc)

                # Provider-wide exhaustion: every remaining level backed by the
                # SAME provider shares the exhausted capacity (e.g. sibling
                # Claude tiers on one subscription), so mark them all visited
                # and skip past them in one step — regardless of
                # max_fallback_depth — instead of wasting fallback hops walking
                # tiers that are already spent. Non-exhaustion failures
                # (transient errors, rate limits) fall back level-by-level.
                if _is_provider_exhausted(exc):
                    exhausted_provider = tlc.provider
                    skipped = _levels_on_provider(tier_config, exhausted_provider)
                    visited |= skipped
                    log.warning(
                        "%s: %s exhausted provider %r — skipping all remaining "
                        "%r-backed tiers (%s)",
                        what,
                        current_level.value,
                        exhausted_provider,
                        exhausted_provider,
                        ", ".join(sorted(lvl.value for lvl in skipped)),
                    )

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

                if run_span is not None:
                    run_span.set_attribute(_ATTR_TIER_PROMOTIONS, promotions + 1)
                    run_span.set_attribute(_ATTR_TIER_FALLBACK_ACTIVATED, True)

                visited.add(next_level)
                promotions += 1
                current_level = next_level
                continue

            if span is not None:
                span.set_attribute(_ATTR_TIER_SUCCEEDED, True)

            # Clear cooldown state on success.
            health_tracker.record_success(tlc.model)

            log.info("%s: %s succeeded", what, current_level.value)
            return result


def call_with_tier_fallback[T](
    fn_factory: Callable[[TierLevelConfig], Callable[[], T]],
    *,
    tier_config: TierConfig,
    level: TierLevel = TierLevel.LEVEL1,
    fallback_enabled: bool = True,
    max_fallback_depth: int = 2,
    what: str = "model call",
    sleep: Callable[[float], None] = time.sleep,
) -> T:
    """Run *fn_factory*-produced callables with tier escalation.

    Args:
        fn_factory: Called with the :class:`TierLevelConfig` for the
            current tier level; must return a zero-argument callable
            that performs the actual work (typically a model call with
            provider-specific retry). Called **fresh** for each tier
            level visited.
        tier_config: A validated :class:`TierConfig` providing
            ``level1``, ``level2``, ``level3``, ``level4``
            :class:`TierLevelConfig` attributes.
        level: The starting :class:`TierLevel` (default ``LEVEL1``).
        fallback_enabled: When ``True`` (the default), escalation
            proceeds according to *max_fallback_depth*. Pass ``False``
            to try only the starting level, re-raising any failure
            immediately — the right choice when a caller depends on a
            specific tier's behaviour rather than on getting an answer.
        max_fallback_depth: Maximum number of tier promotions (tier
            switches) allowed. Default ``2`` means at most two
            promotions, so up to three tiers can be tried. ``0`` means
            no escalation.
        what: Human-readable label for log messages (default ``"model
            call"``).
        sleep: Injectable sleep for tests (default :func:`time.sleep`;
            reserved for future backoff between tier promotions).

    """

    async def _invoke(f: Callable[[], Any]) -> Any:
        return f()

    return _drive_sync(  # type: ignore[no-any-return]
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
    fallback_enabled: bool = True,
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

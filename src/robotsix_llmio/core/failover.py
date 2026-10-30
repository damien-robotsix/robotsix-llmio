"""Provider failover — automatic cascade down the ordered provider-slot chain.

Capability levels never fall back to one another. When a provider slot of
:attr:`TierConfig.chain` fails in a provider-shaped way, calls run the
*same level* on the next slot of the chain instead (position 0, then 1, then
2, …; the legacy names ``default`` / ``fallback`` alias positions 0 / 1):

- **Per call**: a provider-shaped failure on the active slot retries the
  call on the next slot, cascading until a slot succeeds or every slot of
  the chain has been tried once, so a single request survives provider
  outages. The last slot's error propagates when the chain is exhausted.
- **Sticky windows**: every slot except the last tracks its own health.
  After ``failure_threshold`` consecutive provider-shaped failures on a slot
  (a provider-wide exhaustion arms it immediately), the tracker skips that
  slot for ``window_seconds`` (default 15 minutes) and routes calls straight
  to the next slot whose window is not armed. When a window expires the next
  call probes that slot again; a still-broken slot re-arms a fresh,
  exponentially backed-off window. When an exhaustion error
  names its quota reset ("resets 1:10pm (UTC)", "resets Sep 5, 7pm (UTC)"),
  the window arms until that time instead (clamped, with a little slack) —
  no probing a provider that said when it comes back.

The process-wide :class:`ProviderFailoverTracker` singleton holds this state,
and :func:`get_failover_status` exposes it for consumer UIs (which provider
is active, whether failover is armed, and until when). Both live in
:mod:`robotsix_llmio.core._failover_state` and are re-exported here. The
stateless policy — failure classification (:func:`is_provider_shaped`), the
per-request slot-switch cap and the cost-guard wiring — lives in
:mod:`robotsix_llmio.core._failover_policy` and is re-exported here too. This
module keeps only the call loop and its public entry points.

This module does **not** perform local retries — those remain the
responsibility of the callable produced by *fn_factory* (which typically
wraps :func:`~robotsix_llmio.core.retry.call_with_retry` with
provider-specific transient detection). One *fn_factory* callable's failure —
however many internal retries it spent — counts as one provider failure.
"""

from __future__ import annotations

import logging

# ``random`` is consumed by the tracker's backoff jitter in
# ``_failover_state``; kept importable here so tests can patch
# ``failover.random.random`` as before the split.
import random as random
from collections.abc import Awaitable, Callable
from contextlib import AbstractContextManager
from dataclasses import dataclass
from typing import Any, TypeVar

from robotsix_llmio.config.tier import TierConfig, TierLevelConfig

from ._cost_guard import estimate_request_cost as estimate_request_cost
from ._failover_policy import _ATTR_COST_GUARD_BLOCKED as _ATTR_COST_GUARD_BLOCKED
from ._failover_policy import _ATTR_COST_GUARD_REASON as _ATTR_COST_GUARD_REASON
from ._failover_policy import _attempt_order as _attempt_order
from ._failover_policy import _enforce_cost_guard as _enforce_cost_guard
from ._failover_policy import _is_exhaustion as _is_exhaustion
from ._failover_policy import _max_slot_switches as _max_slot_switches
from ._failover_policy import is_provider_shaped as is_provider_shaped
from ._failover_state import FailoverStatus as FailoverStatus
from ._failover_state import ProviderFailoverTracker as ProviderFailoverTracker
from ._failover_state import _slot_label
from ._failover_state import get_failover_status as get_failover_status
from ._failover_state import get_failover_tracker as get_failover_tracker
from ._failover_state import reset_failover_tracker as reset_failover_tracker
from ._otel import get_recording_span, get_tracer, start_span
from ._reset_hint import _MAX_RESET_WINDOW_SECONDS as _MAX_RESET_WINDOW_SECONDS
from ._reset_hint import _parse_reset_delay as _parse_reset_delay
from .retry import _drive_sync

log = logging.getLogger("robotsix_llmio.failover")

_TRACER_NAME: str = "robotsix_llmio.core.failover"

# OTel span attribute keys for failover observability.
# Centralised constants so a rename is a one-line change.
_ATTR_SLOT = "llmio.failover.slot"
_ATTR_LEVEL = "llmio.tier.level"
_ATTR_PROVIDER = "llmio.tier.provider"
_ATTR_MODEL = "llmio.tier.model"
_ATTR_ATTEMPT_INDEX = "llmio.failover.attempt_index"
_ATTR_SUCCEEDED = "llmio.failover.succeeded"
_ATTR_ACTIVATED = "llmio.failover.activated"
# Retry-cap observability (the tracker's backoff attributes live in
# ``_failover_state``).
_ATTR_SLOT_SWITCHES = "llmio.failover.slot_switches"
_ATTR_MAX_SLOT_SWITCHES = "llmio.failover.max_slot_switches"


# Per-attempt span names, chosen by slot so cost/observability reviews are not
# misled: a primary-tier (chain position 0, ``slot=default``) success is NOT a
# failover, so it gets a span name without "failover" in it; the failover name
# is reserved for genuine escalations (any later chain position).
_SPAN_PRIMARY_ATTEMPT = "llmio.attempt.primary"
_SPAN_FAILOVER_ATTEMPT = "llmio.failover.attempt"

T = TypeVar("T")


# --------------------------------------------------------------------------- #
#  The failover call loop                                                     #
# --------------------------------------------------------------------------- #


def _attempt_span(
    level: int, slot_index: int, attempt_index: int, tlc: TierLevelConfig
) -> AbstractContextManager[Any]:
    """Open the per-attempt span, named by slot (primary vs failover)."""
    span_name = _SPAN_PRIMARY_ATTEMPT if slot_index == 0 else _SPAN_FAILOVER_ATTEMPT
    return start_span(
        get_tracer(_TRACER_NAME),
        span_name,
        attributes={
            _ATTR_SLOT: _slot_label(slot_index),
            _ATTR_LEVEL: level,
            _ATTR_PROVIDER: tlc.provider,
            _ATTR_MODEL: tlc.model_name,
            _ATTR_ATTEMPT_INDEX: attempt_index,
        },
    )


def _should_cascade(
    exc: BaseException,
    tracker: ProviderFailoverTracker,
    slot_index: int,
    *,
    last_attempt: bool,
) -> bool:
    """Record a provider-shaped failure; True when the request should cascade.

    Task-shaped failures and failures on the last attempted slot propagate.
    """
    provider_shaped = is_provider_shaped(exc)
    if provider_shaped:
        tracker.record_failure(slot_index, exc)
    return provider_shaped and not last_attempt


def _report_cascade(
    exc: BaseException,
    *,
    slot_name: object,
    candidate_index: int,
    slot_switches: int,
    run_span: Any,
    what: str,
) -> None:
    """Log and record on the run span a cascade to the next slot."""
    log.warning(
        "%s: slot=%s failed with %s — cascading to slot=%s",
        what,
        slot_name,
        type(exc).__name__,
        _slot_label(candidate_index),
    )
    if run_span is not None:
        run_span.set_attribute(_ATTR_ACTIVATED, True)
        run_span.set_attribute(_ATTR_SLOT_SWITCHES, slot_switches)


@dataclass(frozen=True, slots=True)
class _FailoverRun:
    """Per-request context shared by the loop's helpers."""

    tier_config: TierConfig
    level: int
    what: str
    tracker: ProviderFailoverTracker
    order: list[int]
    run_span: Any
    # Cost guards compare every candidate against the slot the request
    # started on ("staying put"), so a long cascade cannot ratchet the price
    # up one tolerable hop at a time.
    primary_tlc: TierLevelConfig


def _start_run(
    tier_config: TierConfig, *, level: int, failover_enabled: bool, what: str
) -> _FailoverRun:
    """Adopt the failover policy and plan the slots this request will try."""
    tracker = get_failover_tracker()
    chain_length = len(tier_config.chain)
    tracker.configure(tier_config.failover, chain_length=chain_length)

    first = tracker.active_index(chain_length=chain_length)
    run_span = get_recording_span()
    if run_span is not None:
        run_span.set_attribute(
            _ATTR_MAX_SLOT_SWITCHES, _max_slot_switches(chain_length)
        )
    return _FailoverRun(
        tier_config=tier_config,
        level=level,
        what=what,
        tracker=tracker,
        order=_attempt_order(first, chain_length, failover_enabled),
        run_span=run_span,
        primary_tlc=tier_config.for_level(level, slot=first),
    )


def _switch_slot(
    run: _FailoverRun,
    exc: BaseException,
    *,
    attempt_index: int,
    slot_switches: int,
) -> None:
    """Cost-guard and report the switch from *attempt_index* to the next slot.

    Raises :class:`FailoverCostGuardError` when the cost guard refuses it.
    """
    slot_name = _slot_label(run.order[attempt_index])
    candidate_index = run.order[attempt_index + 1]
    _enforce_cost_guard(
        exc,
        primary_tlc=run.primary_tlc,
        tier_config=run.tier_config,
        level=run.level,
        slot_name=slot_name,
        candidate_slot=_slot_label(candidate_index),
        run_span=run.run_span,
        what=run.what,
    )
    _report_cascade(
        exc,
        slot_name=slot_name,
        candidate_index=candidate_index,
        slot_switches=slot_switches,
        run_span=run.run_span,
        what=run.what,
    )


async def _failover_loop(
    fn_factory: Callable[[TierLevelConfig], Callable[[], Any]],
    *,
    invoke: Callable[[Callable[[], Any]], Awaitable[Any]],
    tier_config: TierConfig,
    level: int,
    failover_enabled: bool = True,
    what: str = "model call",
) -> Any:
    """Shared failover loop — ``invoke`` adapts sync vs async.

    Starts on the tracker's active chain position and, on each
    provider-shaped failure, cascades to the next slot in chain order
    (wrapping past the end back to the earlier positions), trying every slot
    at most once. The last attempted slot's error propagates unchanged.
    """
    run = _start_run(
        tier_config, level=level, failover_enabled=failover_enabled, what=what
    )
    slot_switches = 0
    for attempt_index, slot_index in enumerate(run.order):
        slot_name = _slot_label(slot_index)
        tlc = tier_config.for_level(level, slot=slot_index)
        with _attempt_span(level, slot_index, attempt_index, tlc) as span:
            log.info(
                "%s: trying slot=%s (level=%d, provider=%s, model=%s)",
                what,
                slot_name,
                level,
                tlc.provider,
                tlc.model_name,
            )
            try:
                result = await invoke(fn_factory(tlc))
            except Exception as exc:
                if span is not None:
                    span.set_attribute(_ATTR_SUCCEEDED, False)
                last_attempt = attempt_index == len(run.order) - 1
                if not _should_cascade(
                    exc, run.tracker, slot_index, last_attempt=last_attempt
                ):
                    raise
                slot_switches += 1
                _switch_slot(
                    run, exc, attempt_index=attempt_index, slot_switches=slot_switches
                )
                continue

            if span is not None:
                span.set_attribute(_ATTR_SUCCEEDED, True)
            run.tracker.record_success(slot_index)
            log.info("%s: slot=%s succeeded", what, slot_name)
            return result

    raise AssertionError("unreachable: the loop returns or raises")


def call_with_failover[T](
    fn_factory: Callable[[TierLevelConfig], Callable[[], T]],
    *,
    tier_config: TierConfig,
    level: int,
    failover_enabled: bool = True,
    what: str = "model call",
) -> T:
    """Run *fn_factory*-produced callables with provider failover.

    Args:
        fn_factory: Called with the :class:`TierLevelConfig` resolved for
            the attempted provider slot; must return a zero-argument
            callable that performs the actual work (typically a model call
            with provider-specific retry). Called **fresh** per slot
            attempted.
        tier_config: A validated :class:`TierConfig`; its ``failover``
            policy is adopted by the process-wide tracker.
        level: Capability level (1, 2, or 3). The level never changes
            across attempts — only the provider slot does.
        failover_enabled: When ``True`` (the default), a provider-shaped
            failure on the active slot retries the same level on the next
            slot of the chain, cascading until one succeeds or the chain is
            exhausted. Pass ``False`` where a caller depends on one specific
            provider's behaviour rather than on getting an answer.
        what: Human-readable label for log messages.

    """

    async def _invoke(f: Callable[[], Any]) -> Any:
        return f()

    return _drive_sync(  # type: ignore[no-any-return]
        _failover_loop(
            fn_factory,
            invoke=_invoke,
            tier_config=tier_config,
            level=level,
            failover_enabled=failover_enabled,
            what=what,
        )
    )


async def acall_with_failover[T](
    fn_factory: Callable[[TierLevelConfig], Callable[[], Awaitable[T]]],
    *,
    tier_config: TierConfig,
    level: int,
    failover_enabled: bool = True,
    what: str = "model call",
) -> T:
    """Async mirror of :func:`call_with_failover`.

    Identical semantics except *fn_factory* returns an awaitable-producing
    callable and the callable is awaited.
    """

    async def _invoke(f: Callable[[], Any]) -> Any:
        return await f()

    return await _failover_loop(  # type: ignore[no-any-return]
        fn_factory,
        invoke=_invoke,
        tier_config=tier_config,
        level=level,
        failover_enabled=failover_enabled,
        what=what,
    )

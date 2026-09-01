"""Provider failover — automatic default→fallback slot switching.

Capability levels never fall back to one another. When the ``default``
provider slot (baked: Anthropic via the Claude SDK) fails in a
provider-shaped way, calls run the *same level* on the ``fallback`` slot
(baked: DeepSeek via OpenRouter) instead:

- **Per call**: a provider-shaped failure on the active slot retries the
  call once on the other slot, so a single request survives a provider
  outage.
- **Sticky window**: after ``failure_threshold`` consecutive provider-shaped
  failures on the ``default`` slot (a provider-wide exhaustion arms it
  immediately), the tracker routes *all* calls straight to ``fallback`` for
  ``window_seconds`` (default 15 minutes), skipping the doomed default
  attempt. When the window expires the next call probes ``default`` again; a
  still-broken default re-arms a fresh window.

The process-wide :class:`ProviderFailoverTracker` singleton holds this state,
and :func:`get_failover_status` exposes it for consumer UIs (which provider
is active, whether failover is armed, and until when).

This module does **not** perform local retries — those remain the
responsibility of the callable produced by *fn_factory* (which typically
wraps :func:`~robotsix_llmio.core.retry.call_with_retry` with
provider-specific transient detection). One *fn_factory* callable's failure —
however many internal retries it spent — counts as one provider failure.
"""

from __future__ import annotations

import logging
import threading
import time
from collections.abc import Awaitable, Callable
from datetime import UTC, datetime, timedelta
from typing import Any, TypeVar

from pydantic import BaseModel

from robotsix_llmio.config.tier import (
    FailoverConfig,
    ProviderSlotName,
    TierConfig,
    TierLevelConfig,
)
from robotsix_llmio.exceptions import ProviderExhaustedError

from ._otel import get_recording_span, get_tracer, start_span
from .retry import (
    _drive_sync,
    _walk_cause_chain,
    is_rate_limited,
    is_transient,
    is_usage_exhausted,
)

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

T = TypeVar("T")


# --------------------------------------------------------------------------- #
#  Failure classification                                                     #
# --------------------------------------------------------------------------- #


def is_provider_shaped(exc: BaseException) -> bool:
    """True when *exc* points at the provider rather than the task.

    A provider-shaped failure is one where running the same request on a
    *different provider* can plausibly succeed: transient transport errors
    that outlived their retries, rate limits, usage exhaustion, and
    provider-wide exhaustion (:class:`ProviderExhaustedError`). Task-shaped
    failures — a turn-limit blowout, a validation error in the caller's own
    output type — raise straight through: re-running a doomed task on the
    other provider would only spend money twice.
    """
    if any(isinstance(cur, ProviderExhaustedError) for cur in _walk_cause_chain(exc)):
        return True
    return is_transient(exc) or is_rate_limited(exc) or is_usage_exhausted(exc)


#: Exception class names that doom every future call on the provider until an
#: external fix, matched by name so the lightweight core does not import the
#: claude_sdk package. ``ClaudeSDKAuthError``: a dead OAuth credential fails
#: every call on the subscription exactly like exhaustion does — waiting for
#: the consecutive-failure threshold just burns doomed attempts.
_PROVIDER_DEAD_NAMES: frozenset[str] = frozenset({"ClaudeSDKAuthError"})


def _is_exhaustion(exc: BaseException) -> bool:
    """True when *exc* signals provider-wide exhaustion or a dead credential.

    Either way every future call on the same provider is also doomed until an
    external fix (quota reset, credential refresh), so failover arms
    immediately instead of waiting for the consecutive-failure threshold.
    """
    for cur in _walk_cause_chain(exc):
        if isinstance(cur, ProviderExhaustedError):
            return True
        if type(cur).__name__ in _PROVIDER_DEAD_NAMES:
            return True
    return is_usage_exhausted(exc)


# --------------------------------------------------------------------------- #
#  Status — the consumer-UI surface                                           #
# --------------------------------------------------------------------------- #


class FailoverStatus(BaseModel):
    """Snapshot of the failover tracker, shaped for consumer status APIs.

    Consumers expose this (e.g. ``.model_dump(mode="json")``) from their own
    status endpoints so their UIs can show which provider slot is serving
    calls and, during failover, when the default slot returns.
    """

    active_slot: ProviderSlotName
    failover_active: bool
    failover_until: datetime | None
    seconds_remaining: float | None
    consecutive_failures: int
    failure_threshold: int
    window_seconds: float
    last_failure_at: datetime | None
    last_failure_reason: str | None


# --------------------------------------------------------------------------- #
#  ProviderFailoverTracker                                                    #
# --------------------------------------------------------------------------- #


class ProviderFailoverTracker:
    """Track default-slot health and route calls between provider slots.

    Thread-safe: consumers run agents from worker threads and event loops
    concurrently, and status reads come from web handlers.
    """

    def __init__(self, config: FailoverConfig | None = None) -> None:
        self._lock = threading.Lock()
        self._config = config if config is not None else FailoverConfig()
        self._consecutive_failures = 0
        #: Monotonic deadline until which calls route to the fallback slot.
        self._failover_until_monotonic = 0.0
        #: Wall-clock twin of the deadline, for status display only.
        self._failover_until_wall: datetime | None = None
        self._last_failure_at: datetime | None = None
        self._last_failure_reason: str | None = None

    # -- routing ------------------------------------------------------------

    def configure(self, config: FailoverConfig) -> None:
        """Adopt *config* as the active threshold + window policy."""
        with self._lock:
            self._config = config

    def active_slot(self, now: float | None = None) -> ProviderSlotName:
        """Return the slot new calls should resolve against."""
        if now is None:
            now = time.monotonic()
        with self._lock:
            if now < self._failover_until_monotonic:
                return "fallback"
            return "default"

    # -- health recording ---------------------------------------------------

    def record_failure(
        self,
        slot: ProviderSlotName,
        exc: BaseException,
        now: float | None = None,
        wall_now: datetime | None = None,
    ) -> None:
        """Record a provider-shaped failure on *slot*.

        Only ``default``-slot failures drive routing: exhaustion arms the
        failover window immediately, other failures arm it once
        ``failure_threshold`` consecutive ones accumulate. Fallback-slot
        failures update the last-failure fields for status display only —
        the window governs the return to ``default``, not fallback health.
        """
        if now is None:
            now = time.monotonic()
        if wall_now is None:
            wall_now = datetime.now(UTC)

        reason = f"{type(exc).__name__}: {exc}"[:500]
        with self._lock:
            self._last_failure_at = wall_now
            self._last_failure_reason = reason
            if slot != "default":
                return

            if _is_exhaustion(exc):
                self._arm_locked(now, wall_now)
                log.warning(
                    "default provider slot exhausted — failover armed for %.0fs: %s",
                    self._config.window_seconds,
                    reason,
                )
                return

            self._consecutive_failures += 1
            if self._consecutive_failures >= self._config.failure_threshold:
                self._arm_locked(now, wall_now)
                log.warning(
                    "default provider slot failed %d consecutive times — "
                    "failover armed for %.0fs: %s",
                    self._config.failure_threshold,
                    self._config.window_seconds,
                    reason,
                )

    def record_success(self, slot: ProviderSlotName) -> None:
        """Record a success on *slot*.

        A ``default``-slot success clears the failure streak and any armed
        window (the provider is demonstrably healthy again). Fallback-slot
        successes change nothing — the window alone decides when calls
        return to ``default``.
        """
        if slot != "default":
            return
        with self._lock:
            self._consecutive_failures = 0
            self._failover_until_monotonic = 0.0
            self._failover_until_wall = None

    def _arm_locked(self, now: float, wall_now: datetime) -> None:
        """Arm the failover window. Caller holds the lock."""
        window = self._config.window_seconds
        self._failover_until_monotonic = now + window
        self._failover_until_wall = wall_now + timedelta(seconds=window)
        self._consecutive_failures = 0

    # -- status -------------------------------------------------------------

    def status(self, now: float | None = None) -> FailoverStatus:
        """Return a :class:`FailoverStatus` snapshot."""
        if now is None:
            now = time.monotonic()
        with self._lock:
            active = now < self._failover_until_monotonic
            return FailoverStatus(
                active_slot="fallback" if active else "default",
                failover_active=active,
                failover_until=self._failover_until_wall if active else None,
                seconds_remaining=(
                    self._failover_until_monotonic - now if active else None
                ),
                consecutive_failures=self._consecutive_failures,
                failure_threshold=self._config.failure_threshold,
                window_seconds=self._config.window_seconds,
                last_failure_at=self._last_failure_at,
                last_failure_reason=self._last_failure_reason,
            )

    def reset(self) -> None:
        """Clear all state (useful for tests)."""
        with self._lock:
            self._consecutive_failures = 0
            self._failover_until_monotonic = 0.0
            self._failover_until_wall = None
            self._last_failure_at = None
            self._last_failure_reason = None


# --------------------------------------------------------------------------- #
#  Module-level singleton                                                     #
# --------------------------------------------------------------------------- #

_tracker: ProviderFailoverTracker | None = None
_tracker_lock = threading.Lock()


def get_failover_tracker() -> ProviderFailoverTracker:
    """Return the process-wide :class:`ProviderFailoverTracker` singleton."""
    global _tracker
    with _tracker_lock:
        if _tracker is None:
            _tracker = ProviderFailoverTracker()
        return _tracker


def reset_failover_tracker() -> None:
    """Reset the process-wide singleton (for test teardown)."""
    global _tracker
    with _tracker_lock:
        _tracker = None


def get_failover_status() -> FailoverStatus:
    """Snapshot the process-wide failover state — the consumer-UI surface."""
    return get_failover_tracker().status()


# --------------------------------------------------------------------------- #
#  The failover call loop                                                     #
# --------------------------------------------------------------------------- #


async def _failover_loop(
    fn_factory: Callable[[TierLevelConfig], Callable[[], Any]],
    *,
    invoke: Callable[[Callable[[], Any]], Awaitable[Any]],
    tier_config: TierConfig,
    level: int,
    failover_enabled: bool = True,
    what: str = "model call",
) -> Any:
    """Shared failover loop — ``invoke`` adapts sync vs async."""
    tracker = get_failover_tracker()
    tracker.configure(tier_config.failover)

    first: ProviderSlotName = tracker.active_slot()
    order: list[ProviderSlotName] = [first]
    if failover_enabled:
        order.append("fallback" if first == "default" else "default")

    run_span = get_recording_span()

    for attempt_index, slot_name in enumerate(order):
        tlc = tier_config.for_level(level, slot=slot_name)

        with start_span(
            get_tracer(_TRACER_NAME),
            "llmio.failover.attempt",
            attributes={
                _ATTR_SLOT: slot_name,
                _ATTR_LEVEL: level,
                _ATTR_PROVIDER: tlc.provider,
                _ATTR_MODEL: tlc.model_name,
                _ATTR_ATTEMPT_INDEX: attempt_index,
            },
        ) as span:
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

                provider_shaped = is_provider_shaped(exc)
                if provider_shaped:
                    tracker.record_failure(slot_name, exc)

                last_attempt = attempt_index == len(order) - 1
                if last_attempt or not provider_shaped:
                    raise

                log.warning(
                    "%s: slot=%s failed with %s — retrying on the other slot",
                    what,
                    slot_name,
                    type(exc).__name__,
                )
                if run_span is not None:
                    run_span.set_attribute(_ATTR_ACTIVATED, True)
                continue

            if span is not None:
                span.set_attribute(_ATTR_SUCCEEDED, True)
            tracker.record_success(slot_name)
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
            failure on the active slot retries the same level on the other
            slot. Pass ``False`` where a caller depends on one specific
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

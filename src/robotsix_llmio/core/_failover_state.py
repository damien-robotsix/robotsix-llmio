"""Failover state — per-slot health, the tracker, and its process-wide singleton.

Extracted from :mod:`robotsix_llmio.core.failover`, which re-exports the
public names (:class:`ProviderFailoverTracker`, :class:`FailoverStatus`,
:func:`get_failover_tracker`, :func:`reset_failover_tracker`,
:func:`get_failover_status`) so ``robotsix_llmio.core.failover`` stays the
import surface. This module holds the mutable routing state only; the call
loop that consults it lives in :mod:`robotsix_llmio.core.failover`, and the
stateless classification it relies on (:func:`_is_exhaustion`) in
:mod:`robotsix_llmio.core._failover_policy`.
"""

from __future__ import annotations

import logging
import random
import threading
import time
from datetime import UTC, datetime, timedelta
from typing import cast

from pydantic import BaseModel

from robotsix_llmio.config.tier import PROVIDER_SLOTS, FailoverConfig, ProviderSlotName

from ._failover_policy import _is_exhaustion
from ._otel import get_recording_span
from ._reset_hint import (
    _MAX_RESET_WINDOW_SECONDS,
    _RESET_SLACK_SECONDS,
    _parse_reset_delay,
)

log = logging.getLogger("robotsix_llmio.failover")

# OTel span attribute keys for failover-window backoff observability.
_ATTR_REATTEMPT_COUNT = "llmio.failover.reattempt_count"
_ATTR_BACKOFF_SECONDS = "llmio.failover.backoff_seconds"


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
#  Backoff telemetry                                                          #
# --------------------------------------------------------------------------- #


def _record_backoff_span(reattempt_count: int, window_seconds: float) -> None:
    """Record failover-window backoff metrics on the current OTel span.

    No-op when no span is recording (mirrors the retry module's
    ``_record_transient_retry_span``). *reattempt_count* is 1 on the first
    arm since the last default success and grows by one per consecutive
    re-arm; *window_seconds* is the (possibly backed-off) window actually
    applied before the degraded primary is re-probed.
    """
    span = get_recording_span()
    if span is None:
        return
    span.set_attribute(_ATTR_REATTEMPT_COUNT, reattempt_count)
    span.set_attribute(_ATTR_BACKOFF_SECONDS, window_seconds)


# --------------------------------------------------------------------------- #
#  ProviderFailoverTracker                                                    #
# --------------------------------------------------------------------------- #


def _slot_index(slot: ProviderSlotName) -> int:
    """Map a slot reference (chain index or legacy alias) to its chain index."""
    if isinstance(slot, str):
        return PROVIDER_SLOTS.index(slot)
    return slot


def _slot_label(index: int) -> ProviderSlotName:
    """Return the public label for chain position *index*.

    The first two positions keep their legacy names (``"default"`` /
    ``"fallback"``) so two-slot consumers — status UIs, span filters on
    ``llmio.failover.slot`` — see exactly what they saw before the chain was
    generalised; deeper positions are reported by their integer index.
    """
    if index < len(PROVIDER_SLOTS):
        return cast(ProviderSlotName, PROVIDER_SLOTS[index])
    return index


class _SlotHealth:
    """Mutable per-slot health state. Guarded by the tracker's lock."""

    __slots__ = (
        "consecutive_failures",
        "reattempt_count",
        "until_monotonic",
        "until_wall",
    )

    def __init__(self) -> None:
        self.consecutive_failures = 0
        #: Monotonic deadline until which calls skip this slot.
        self.until_monotonic = 0.0
        #: Wall-clock twin of the deadline, for status display only.
        self.until_wall: datetime | None = None
        #: Number of times this slot's window has armed since its last
        #: success. 1 after the first arm, incrementing per re-arm; drives the
        #: exponential backoff that delays re-probing a persistently degraded
        #: slot. Reset to 0 whenever the slot succeeds.
        self.reattempt_count = 0

    def armed(self, now: float) -> bool:
        return now < self.until_monotonic


class ProviderFailoverTracker:
    """Track per-slot health and route calls down the provider chain.

    Every slot of the ordered chain (see :attr:`TierConfig.chain`) has its
    own failure streak and failover window. New calls resolve against the
    *first* slot whose window is not armed, so failures cascade down the
    chain — position 0, then 1, then 2, … — and calls climb back up as each
    earlier slot's window expires (the next call then probes it). The last
    slot never arms: there is nothing further to cascade to, so its failures
    only update the last-failure status fields.

    With a two-slot chain this is exactly the historic ``default`` /
    ``fallback`` behaviour.

    Thread-safe: consumers run agents from worker threads and event loops
    concurrently, and status reads come from web handlers.
    """

    def __init__(self, config: FailoverConfig | None = None) -> None:
        self._lock = threading.Lock()
        self._config = config if config is not None else FailoverConfig()
        #: Length of the chain being routed, adopted via :meth:`configure`.
        #: ``None`` until a chain is known; the legacy two-slot shape is
        #: assumed for arming decisions in that case.
        self._chain_length: int | None = None
        self._health: dict[int, _SlotHealth] = {}
        self._last_failure_at: datetime | None = None
        self._last_failure_reason: str | None = None

    # -- routing ------------------------------------------------------------

    def configure(
        self, config: FailoverConfig, chain_length: int | None = None
    ) -> None:
        """Adopt *config* as the active threshold + window policy.

        *chain_length*, when given, is the number of slots in the routed
        chain: it bounds :meth:`active_slot` and decides which slot is last
        (and therefore never arms).
        """
        with self._lock:
            self._config = config
            if chain_length is not None:
                self._chain_length = chain_length

    def _health_locked(self, index: int) -> _SlotHealth:
        health = self._health.get(index)
        if health is None:
            health = self._health[index] = _SlotHealth()
        return health

    def _active_index_locked(self, now: float, chain_length: int | None) -> int:
        """First chain position whose window is not armed. Caller holds lock."""
        if chain_length is None:
            chain_length = self._chain_length
        index = 0
        while (health := self._health.get(index)) is not None and health.armed(now):
            index += 1
        if chain_length is not None:
            index = min(index, chain_length - 1)
        return index

    def active_index(
        self, now: float | None = None, chain_length: int | None = None
    ) -> int:
        """Return the zero-based chain position new calls should start on.

        *chain_length* clamps the result to the last slot of the routed
        chain (defaults to the length adopted via :meth:`configure`).
        """
        if now is None:
            now = time.monotonic()
        with self._lock:
            return self._active_index_locked(now, chain_length)

    def active_slot(
        self, now: float | None = None, chain_length: int | None = None
    ) -> ProviderSlotName:
        """Return the slot new calls should resolve against.

        Positions 0 and 1 are reported by their legacy names (``"default"``
        / ``"fallback"``), deeper positions by their integer index.
        """
        return _slot_label(self.active_index(now, chain_length))

    # -- health recording ---------------------------------------------------

    def _is_last_locked(self, index: int) -> bool:
        chain_length = self._chain_length
        if chain_length is None:
            chain_length = len(PROVIDER_SLOTS)
        return index >= chain_length - 1

    def record_failure(
        self,
        slot: ProviderSlotName,
        exc: BaseException,
        now: float | None = None,
        wall_now: datetime | None = None,
    ) -> None:
        """Record a provider-shaped failure on *slot*.

        Exhaustion arms that slot's failover window immediately, other
        failures arm it once ``failure_threshold`` consecutive ones
        accumulate; while armed, calls cascade to the next slot. Failures on
        the last slot of the chain update the last-failure fields for status
        display only — there is nowhere further to cascade, and the earlier
        slots' windows govern the climb back up the chain.
        """
        if now is None:
            now = time.monotonic()
        if wall_now is None:
            wall_now = datetime.now(UTC)

        index = _slot_index(slot)
        reason = f"{type(exc).__name__}: {exc}"[:500]
        with self._lock:
            self._last_failure_at = wall_now
            self._last_failure_reason = reason
            if self._is_last_locked(index):
                return
            health = self._health_locked(index)

            if _is_exhaustion(exc):
                self._handle_exhaustion_locked(
                    health, index, exc, now, wall_now, reason
                )
                return

            health.consecutive_failures += 1
            if health.consecutive_failures >= self._config.failure_threshold:
                window = self._arm_locked(health, now, wall_now)
                log.warning(
                    "provider slot=%s failed %d consecutive times — "
                    "failover armed for %.0fs: %s",
                    _slot_label(index),
                    self._config.failure_threshold,
                    window,
                    reason,
                )

    def _handle_exhaustion_locked(
        self,
        health: _SlotHealth,
        index: int,
        exc: BaseException,
        now: float,
        wall_now: datetime,
        reason: str,
    ) -> None:
        """Arm *health*'s failover window for a provider-wide exhaustion.

        Caller holds the lock. A parsed quota-reset hint arms the window until
        that time (plus slack, clamped to :data:`_MAX_RESET_WINDOW_SECONDS` and
        floored at the configured window so a stale hint cannot shrink it into a
        tight probe loop); otherwise the fixed ``window_seconds`` applies.
        """
        hinted = _parse_reset_delay(str(exc), wall_now)
        if hinted is not None:
            window = min(hinted + _RESET_SLACK_SECONDS, _MAX_RESET_WINDOW_SECONDS)
            # Never arm SHORTER than the configured window off a hint — a stale
            # "resets 2 minutes from now" would turn failover into a tight probe
            # loop.
            window = max(window, self._config.window_seconds)
        else:
            window = self._config.window_seconds
        # A parsed reset hint is a KNOWN recovery time — never grow or jitter
        # it with reattempt backoff (that would probe before the quota is
        # back, or park the fleet past it). Backoff applies only to the fixed
        # window, which is the degraded-slot churn this caps.
        applied = self._arm_locked(
            health, now, wall_now, window=window, backoff=hinted is None
        )
        log.warning(
            "provider slot=%s exhausted — failover armed for %.0fs: %s",
            _slot_label(index),
            applied,
            reason,
        )

    def record_success(self, slot: ProviderSlotName) -> None:
        """Record a success on *slot*.

        A success clears that slot's failure streak and any armed window (the
        provider is demonstrably healthy again) and forgets its backoff
        escalation. Other slots are untouched — an earlier slot's window
        alone decides when calls climb back to it.
        """
        index = _slot_index(slot)
        with self._lock:
            self._health.pop(index, None)

    def _arm_locked(
        self,
        health: _SlotHealth,
        now: float,
        wall_now: datetime,
        window: float | None = None,
        backoff: bool = True,
    ) -> float:
        """Arm *health*'s failover window and return its length.

        Caller holds the lock. The first arming since the slot's last success
        uses *window* (or the configured ``window_seconds``) unchanged. Each
        subsequent re-arm — a still-degraded slot that failed its probe after
        the window expired — grows the window with exponential backoff and
        jitter (see :meth:`_backoff_window`) when *backoff* is true, so a
        persistently broken slot is re-probed exponentially less often
        instead of churning a doomed attempt every window. Pass
        ``backoff=False`` to honour a deliberate window (e.g. a parsed
        quota-reset hint) verbatim.
        """
        base = self._config.window_seconds if window is None else window
        if backoff and health.reattempt_count > 0:
            window = self._backoff_window(base, health.reattempt_count)
        else:
            window = base
        health.until_monotonic = now + window
        health.until_wall = wall_now + timedelta(seconds=window)
        health.consecutive_failures = 0
        health.reattempt_count += 1
        _record_backoff_span(health.reattempt_count, window)
        return window

    def _backoff_window(self, base: float, reattempt_count: int) -> float:
        """Exponential-backoff-with-jitter window for a repeated reattempt.

        Caller holds the lock; *reattempt_count* is the number of arms
        already recorded on the slot since its last success (>= 1 here). The
        raw window is ``min(base * backoff_base ** reattempt_count, cap)``;
        jitter is then subtracted multiplicatively so the result lands in
        ``[raw * (1 - jitter), raw]``.
        """
        cfg = self._config
        raw = min(
            base * cfg.reattempt_backoff_base**reattempt_count,
            cfg.reattempt_backoff_cap_seconds,
        )
        jitter = raw * cfg.reattempt_backoff_jitter * random.random()  # noqa: S311
        return raw - jitter

    # -- status -------------------------------------------------------------

    def status(self, now: float | None = None) -> FailoverStatus:
        """Return a :class:`FailoverStatus` snapshot.

        ``failover_until`` / ``seconds_remaining`` name the earliest moment
        calls climb back up the chain (the soonest-expiring window among the
        armed slots ahead of the active one).
        """
        if now is None:
            now = time.monotonic()
        with self._lock:
            index = self._active_index_locked(now, None)
            ahead = [h for i, h in self._health.items() if i < index and h.armed(now)]
            soonest = min(ahead, key=lambda h: h.until_monotonic) if ahead else None
            current = self._health.get(index)
            return FailoverStatus(
                active_slot=_slot_label(index),
                failover_active=index > 0,
                failover_until=soonest.until_wall if soonest is not None else None,
                seconds_remaining=(
                    soonest.until_monotonic - now if soonest is not None else None
                ),
                consecutive_failures=(
                    current.consecutive_failures if current is not None else 0
                ),
                failure_threshold=self._config.failure_threshold,
                window_seconds=self._config.window_seconds,
                last_failure_at=self._last_failure_at,
                last_failure_reason=self._last_failure_reason,
            )

    def reset(self) -> None:
        """Clear all state (useful for tests)."""
        with self._lock:
            self._health.clear()
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

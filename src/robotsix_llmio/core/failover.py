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
is active, whether failover is armed, and until when).

This module does **not** perform local retries — those remain the
responsibility of the callable produced by *fn_factory* (which typically
wraps :func:`~robotsix_llmio.core.retry.call_with_retry` with
provider-specific transient detection). One *fn_factory* callable's failure —
however many internal retries it spent — counts as one provider failure.
"""

from __future__ import annotations

import logging
import random
import re
import threading
import time
from collections.abc import Awaitable, Callable
from datetime import UTC, datetime, timedelta
from typing import Any, TypeVar, cast

from pydantic import BaseModel

from robotsix_llmio.config.tier import (
    PROVIDER_SLOTS,
    FailoverConfig,
    ProviderSlotName,
    TierConfig,
    TierLevelConfig,
)
from robotsix_llmio.exceptions import FailoverCostGuardError, ProviderExhaustedError

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
# Retry-cap + backoff observability.
_ATTR_SLOT_SWITCHES = "llmio.failover.slot_switches"
_ATTR_MAX_SLOT_SWITCHES = "llmio.failover.max_slot_switches"
_ATTR_REATTEMPT_COUNT = "llmio.failover.reattempt_count"
_ATTR_BACKOFF_SECONDS = "llmio.failover.backoff_seconds"
# Per-request cost-guard observability: set when a slot switch is refused
# because the candidate slot would cost too much.
_ATTR_COST_GUARD_BLOCKED = "llmio.failover.cost_guard_blocked"
_ATTR_COST_GUARD_REASON = "llmio.failover.cost_guard_reason"


def _max_slot_switches(chain_length: int) -> int:
    """Hard cap on provider-slot switches within a SINGLE request.

    A request cascades through the chain trying each slot AT MOST once, so it
    switches slots at most ``chain_length - 1`` times (exactly once for the
    legacy two-slot chain). This bounds per-request churn so a doomed request
    can never ping-pong between slots, multiplying spend on duplicate prompts.
    """
    return max(chain_length - 1, 0)


# Per-attempt span names, chosen by slot so cost/observability reviews are not
# misled: a primary-tier (chain position 0, ``slot=default``) success is NOT a
# failover, so it gets a span name without "failover" in it; the failover name
# is reserved for genuine escalations (any later chain position).
_SPAN_PRIMARY_ATTEMPT = "llmio.attempt.primary"
_SPAN_FAILOVER_ATTEMPT = "llmio.failover.attempt"

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
#  Reset-hint parsing                                                         #
# --------------------------------------------------------------------------- #

# The Claude CLI names the quota reset inside the exhaustion text, in two
# shapes: the rolling cap ("resets 1:10pm (UTC)", same day or tomorrow) and
# the weekly cap ("resets Sep 5, 7pm (UTC)"). When a hint parses, the
# failover window arms until that time instead of the fixed
# ``window_seconds`` — probing a provider that TOLD us when it comes back
# just burns a failed CLI spawn every window and makes status UIs blink.
_RESET_HINT_RE = re.compile(
    r"resets\s+(?:(?P<month>[a-z]{3,9})\s+(?P<day>\d{1,2}),?\s+)?"
    r"(?P<hour>\d{1,2})(?::(?P<minute>\d{2}))?\s*(?P<meridiem>am|pm)\s*\(utc\)",
    re.IGNORECASE,
)

_MONTHS = {
    m: i
    for i, names in enumerate(
        (
            ("jan", "january"),
            ("feb", "february"),
            ("mar", "march"),
            ("apr", "april"),
            ("may",),
            ("jun", "june"),
            ("jul", "july"),
            ("aug", "august"),
            ("sep", "september"),
            ("oct", "october"),
            ("nov", "november"),
            ("dec", "december"),
        ),
        start=1,
    )
    for m in names
}

#: Defensive ceiling on a hint-derived window: a mis-parsed or hostile
#: "resets" value can never park the fleet on the paid slot for more than a
#: worst-case weekly cycle (+ slack).
_MAX_RESET_WINDOW_SECONDS: float = 8 * 24 * 3600.0

#: Armed windows end this long AFTER the hinted reset, so the first probe
#: lands when the quota is actually back rather than seconds before.
_RESET_SLACK_SECONDS: float = 120.0


def _parse_reset_delay(text: str, wall_now: datetime) -> float | None:
    """Seconds from *wall_now* until the reset named in *text*, or ``None``.

    Handles both CLI shapes: time-only rolls forward to the next occurrence
    (today or tomorrow); a month+day form pins that calendar date (rolling to
    next year when it already passed, e.g. "resets Jan 2" seen in late Dec).
    All times are interpreted as UTC per the ``(UTC)`` marker.
    """
    match = _RESET_HINT_RE.search(text)
    if match is None:
        return None

    hour = int(match.group("hour"))
    minute = int(match.group("minute") or 0)
    if not (1 <= hour <= 12) or minute > 59:
        return None
    hour24 = hour % 12
    if match.group("meridiem").lower() == "pm":
        hour24 += 12

    month_name = match.group("month")
    if month_name is not None:
        month = _MONTHS.get(month_name.lower())
        day = int(match.group("day"))
        if month is None:
            return None
        try:
            target = wall_now.replace(
                month=month,
                day=day,
                hour=hour24,
                minute=minute,
                second=0,
                microsecond=0,
            )
        except ValueError:
            return None
        if target <= wall_now:
            target = target.replace(year=target.year + 1)
    else:
        target = wall_now.replace(hour=hour24, minute=minute, second=0, microsecond=0)
        if target <= wall_now:
            target += timedelta(days=1)

    return (target - wall_now).total_seconds()


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


# --------------------------------------------------------------------------- #
#  Per-request cost guard                                                     #
# --------------------------------------------------------------------------- #

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
    """Shared failover loop — ``invoke`` adapts sync vs async.

    Starts on the tracker's active chain position and, on each
    provider-shaped failure, cascades to the next slot in chain order
    (wrapping past the end back to the earlier positions), trying every slot
    at most once. The last attempted slot's error propagates unchanged.
    """
    tracker = get_failover_tracker()
    chain_length = len(tier_config.chain)
    tracker.configure(tier_config.failover, chain_length=chain_length)

    first = tracker.active_index(chain_length=chain_length)
    order: list[int] = [first]
    if failover_enabled:
        order.extend((first + step) % chain_length for step in range(1, chain_length))
    # Enforce the retry cap explicitly: whatever the order build produced, a
    # single request attempts each slot at most once — it can never switch
    # slots more than ``chain_length - 1`` times, so a doomed request cannot
    # ping-pong between slots.
    max_switches = _max_slot_switches(chain_length)
    order = order[: 1 + max_switches]

    run_span = get_recording_span()
    if run_span is not None:
        run_span.set_attribute(_ATTR_MAX_SLOT_SWITCHES, max_switches)
    slot_switches = 0
    # Cost guards compare every candidate against the slot the request
    # started on ("staying put"), so a long cascade cannot ratchet the price
    # up one tolerable hop at a time.
    primary_tlc = tier_config.for_level(level, slot=first)

    for attempt_index, slot_index in enumerate(order):
        slot_name = _slot_label(slot_index)
        tlc = tier_config.for_level(level, slot=slot_index)

        span_name = _SPAN_PRIMARY_ATTEMPT if slot_index == 0 else _SPAN_FAILOVER_ATTEMPT
        with start_span(
            get_tracer(_TRACER_NAME),
            span_name,
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
                    tracker.record_failure(slot_index, exc)

                last_attempt = attempt_index == len(order) - 1
                if last_attempt or not provider_shaped:
                    raise

                # Cost guard: before spending money retrying on the candidate
                # slot, refuse the switch when that slot would blow the
                # per-request ceiling or be materially pricier than the slot
                # the request started on.
                candidate_slot = _slot_label(order[attempt_index + 1])
                candidate_tlc = tier_config.for_level(level, slot=candidate_slot)
                refusal = _cost_guard_verdict(
                    primary_tlc, candidate_tlc, tier_config.failover
                )
                if refusal is not None:
                    log.warning(
                        "%s: refusing failover slot=%s -> slot=%s: %s",
                        what,
                        slot_name,
                        candidate_slot,
                        refusal,
                    )
                    if run_span is not None:
                        run_span.set_attribute(_ATTR_COST_GUARD_BLOCKED, True)
                        run_span.set_attribute(_ATTR_COST_GUARD_REASON, refusal)
                    raise FailoverCostGuardError(
                        f"{what}: failover to slot={candidate_slot} refused "
                        f"by the per-request cost guard ({refusal})"
                    ) from exc

                slot_switches += 1
                log.warning(
                    "%s: slot=%s failed with %s — cascading to slot=%s",
                    what,
                    slot_name,
                    type(exc).__name__,
                    candidate_slot,
                )
                if run_span is not None:
                    run_span.set_attribute(_ATTR_ACTIVATED, True)
                    run_span.set_attribute(_ATTR_SLOT_SWITCHES, slot_switches)
                continue

            if span is not None:
                span.set_attribute(_ATTR_SUCCEEDED, True)
            tracker.record_success(slot_index)
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

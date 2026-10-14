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
  still-broken default re-arms a fresh window. When an exhaustion error
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
from typing import Any, TypeVar

from pydantic import BaseModel

from robotsix_llmio.config.tier import (
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

#: Hard cap on provider-slot switches within a SINGLE request: a request may
#: try its active slot and fail over to the other slot AT MOST once. This
#: bounds per-request churn so a doomed request can never ping-pong between
#: slots, multiplying spend on duplicate prompts.
_MAX_SLOT_SWITCHES: int = 1

# Per-attempt span names, chosen by slot so cost/observability reviews are not
# misled: a primary-tier (``slot=default``) success is NOT a failover, so it
# gets a span name without "failover" in it; the failover name is reserved for
# genuine escalations (``slot=fallback``).
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
        #: Number of times the window has armed since the last default-slot
        #: success. 1 after the first arm, incrementing per re-arm; drives the
        #: exponential backoff that delays re-probing a persistently degraded
        #: primary. Reset to 0 whenever the default slot succeeds.
        self._reattempt_count = 0

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
                self._handle_exhaustion_locked(exc, now, wall_now, reason)
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

    def _handle_exhaustion_locked(
        self,
        exc: BaseException,
        now: float,
        wall_now: datetime,
        reason: str,
    ) -> None:
        """Arm the failover window for a provider-wide exhaustion.

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
        # window, which is the degraded-primary churn this caps.
        self._arm_locked(now, wall_now, window=window, backoff=hinted is None)
        log.warning(
            "default provider slot exhausted — failover armed for %.0fs%s: %s",
            window,
            " (until the hinted quota reset)" if hinted is not None else "",
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
            # The primary is healthy again — forget the backoff escalation so
            # the next degradation starts from the base window.
            self._reattempt_count = 0

    def _arm_locked(
        self,
        now: float,
        wall_now: datetime,
        window: float | None = None,
        backoff: bool = True,
    ) -> None:
        """Arm the failover window. Caller holds the lock.

        The first arming since the last default-slot success uses *window*
        (or the configured ``window_seconds``) unchanged. Each subsequent
        re-arm — a still-degraded default that failed its probe after the
        window expired — grows the window with exponential backoff and jitter
        (see :meth:`_backoff_window`) when *backoff* is true, so a
        persistently broken primary is re-probed exponentially less often
        instead of churning a doomed attempt every window. Pass
        ``backoff=False`` to honour a deliberate window (e.g. a parsed
        quota-reset hint) verbatim.
        """
        base = self._config.window_seconds if window is None else window
        if backoff and self._reattempt_count > 0:
            window = self._backoff_window(base)
        else:
            window = base
        self._failover_until_monotonic = now + window
        self._failover_until_wall = wall_now + timedelta(seconds=window)
        self._consecutive_failures = 0
        self._reattempt_count += 1
        _record_backoff_span(self._reattempt_count, window)

    def _backoff_window(self, base: float) -> float:
        """Exponential-backoff-with-jitter window for a repeated reattempt.

        Caller holds the lock; ``_reattempt_count`` is the number of arms
        already recorded since the last default success (>= 1 here). The raw
        window is ``min(base * backoff_base ** reattempt_count, cap)``; jitter
        is then subtracted multiplicatively so the result lands in
        ``[raw * (1 - jitter), raw]``.
        """
        cfg = self._config
        raw = min(
            base * cfg.reattempt_backoff_base**self._reattempt_count,
            cfg.reattempt_backoff_cap_seconds,
        )
        jitter = raw * cfg.reattempt_backoff_jitter * random.random()  # noqa: S311
        return raw - jitter

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
            self._reattempt_count = 0


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
    """Shared failover loop — ``invoke`` adapts sync vs async."""
    tracker = get_failover_tracker()
    tracker.configure(tier_config.failover)

    first: ProviderSlotName = tracker.active_slot()
    order: list[ProviderSlotName] = [first]
    if failover_enabled:
        order.append("fallback" if first == "default" else "default")
    # Enforce the one-slot-switch retry cap explicitly: whatever the order
    # build produced, a single request attempts the active slot plus at most
    # ``_MAX_SLOT_SWITCHES`` fallbacks — it can never switch slots more than
    # once, so a doomed request cannot ping-pong between slots.
    order = order[: 1 + _MAX_SLOT_SWITCHES]

    run_span = get_recording_span()
    if run_span is not None:
        run_span.set_attribute(_ATTR_MAX_SLOT_SWITCHES, _MAX_SLOT_SWITCHES)
    slot_switches = 0

    for attempt_index, slot_name in enumerate(order):
        tlc = tier_config.for_level(level, slot=slot_name)

        span_name = (
            _SPAN_PRIMARY_ATTEMPT if slot_name == "default" else _SPAN_FAILOVER_ATTEMPT
        )
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
                    tracker.record_failure(slot_name, exc)

                last_attempt = attempt_index == len(order) - 1
                if last_attempt or not provider_shaped:
                    raise

                # Cost guard: before spending money retrying on the candidate
                # slot, refuse the switch when that slot would blow the
                # per-request ceiling or be materially pricier than this one.
                candidate_slot = order[attempt_index + 1]
                candidate_tlc = tier_config.for_level(level, slot=candidate_slot)
                refusal = _cost_guard_verdict(tlc, candidate_tlc, tier_config.failover)
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
                    "%s: slot=%s failed with %s — retrying on the other slot",
                    what,
                    slot_name,
                    type(exc).__name__,
                )
                if run_span is not None:
                    run_span.set_attribute(_ATTR_ACTIVATED, True)
                    run_span.set_attribute(_ATTR_SLOT_SWITCHES, slot_switches)
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

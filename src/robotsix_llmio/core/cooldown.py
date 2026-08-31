"""Model cooldown tracker — skip known-dead models in the fallback chain.

When a model fails terminally (credit exhaustion, persistent provider error)
a configurable number of consecutive times, the tracker marks it *unhealthy*
and skips it for a cooldown window. After the cooldown expires, one probe
attempt is allowed — success clears the cooldown; another failure re-arms it
for a fresh window.

Both the cooldown duration and the consecutive-failure threshold are
configurable via environment variables:

* ``LLMIO_COOLDOWN_DURATION_SECONDS`` — cooldown window in seconds
  (default: 21600 = 6 hours).
* ``LLMIO_COOLDOWN_FAILURE_THRESHOLD`` — number of consecutive terminal
  failures before the model enters cooldown (default: 3).

Provider-family latch (Claude SDK)
----------------------------------
The Claude tiers (``claudeSDK-*``) all draw on ONE subscription, so once
one is exhausted the rest are too. Waiting for each model to separately
accumulate ``failure_threshold`` failures wastes a probe (~2s) per tier on
every fallback walk. To avoid that, a single usage-exhaustion failure on
any ``claudeSDK-`` model arms cooldown IMMEDIATELY for the WHOLE family:
:meth:`ModelHealthTracker.is_in_cooldown` then returns ``True`` for every
``claudeSDK-`` model until the latch expires. The latch expiry is derived
from the reset hint the error carries (``resets 1:10pm (UTC)``) — clamped
to a 6-hour maximum, falling back to the fixed cooldown duration when the
hint is absent or unparseable. A success on any family model clears the
latch (the quota reset arrived early).
"""

from __future__ import annotations

import os
import re
import time
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta

from .retry import is_usage_exhausted

# ---------------------------------------------------------------------------
# Environment variable names (module-level constants so a rename is one line)
# ---------------------------------------------------------------------------

_ENV_COOLDOWN_DURATION = "LLMIO_COOLDOWN_DURATION_SECONDS"
_ENV_COOLDOWN_THRESHOLD = "LLMIO_COOLDOWN_FAILURE_THRESHOLD"

_DEFAULT_COOLDOWN_DURATION: float = 6 * 3600.0  # 6 hours
_DEFAULT_FAILURE_THRESHOLD: int = 3

# ---------------------------------------------------------------------------
# Provider-family latch (Claude SDK subscription)
# ---------------------------------------------------------------------------

# Every ``claudeSDK-*`` tier draws on the same subscription, so exhaustion on
# one arms cooldown for all of them (see module docstring).
_CLAUDE_SDK_PREFIX = "claudeSDK-"

# Upper bound on a latch derived from a reset hint — a defensive clamp so a
# mis-parsed or hostile "resets" value can never park the family for longer
# than the fixed cooldown window would.
_MAX_RESET_CLAMP_SECONDS: float = 6 * 3600.0  # 6 hours

# Parse the Claude CLI's reset hint, e.g. "resets 1:10pm (UTC)" or
# "resets 8pm (UTC)". Mirrors robotsix-chat's ``claude_usage_reset_at``:
# a 1-2 digit hour, optional ``:MM`` minutes, an am/pm meridiem, and the
# ``(UTC)`` marker. Matched case-insensitively.
_RESET_HINT_RE = re.compile(
    r"resets\s+(\d{1,2})(?::(\d{2}))?\s*(am|pm)\s*\(utc\)",
    re.IGNORECASE,
)


def _family_of(model_id: str) -> str | None:
    """Return the provider-family prefix *model_id* belongs to, or ``None``.

    Currently only the Claude SDK subscription forms a latched family.
    """
    if model_id.startswith(_CLAUDE_SDK_PREFIX):
        return _CLAUDE_SDK_PREFIX
    return None


def _parse_reset_delay(text: str, wall_now: datetime) -> float | None:
    """Return seconds from *wall_now* until the reset time named in *text*.

    Parses a ``resets H[:MM]am|pm (UTC)`` hint. The named time is interpreted
    in UTC; if it has already passed today it rolls over to the next day.
    Returns ``None`` when no valid hint is present.
    """
    match = _RESET_HINT_RE.search(text)
    if match is None:
        return None

    hour = int(match.group(1))
    minute = int(match.group(2)) if match.group(2) is not None else 0
    meridiem = match.group(3).lower()
    if not (1 <= hour <= 12) or minute > 59:
        return None

    hour24 = hour % 12
    if meridiem == "pm":
        hour24 += 12

    target = wall_now.replace(hour=hour24, minute=minute, second=0, microsecond=0)
    if target <= wall_now:
        target += timedelta(days=1)
    return (target - wall_now).total_seconds()


def _env_float(name: str, default: float) -> float:
    raw = os.environ.get(name)
    if raw is None:
        return default
    try:
        return float(raw)
    except ValueError:
        return default


def _env_int(name: str, default: int) -> int:
    raw = os.environ.get(name)
    if raw is None:
        return default
    try:
        return int(raw)
    except ValueError:
        return default


# ---------------------------------------------------------------------------
# ModelHealthState — per-model tracking
# ---------------------------------------------------------------------------


@dataclass
class ModelHealthState:
    """Per-model health tracking state.

    Attributes:
        consecutive_failures: Running count of consecutive terminal failures.
        cooldown_until: Monotonic timestamp (``time.monotonic()``) until which
            the model is skipped.  ``0.0`` means not in cooldown.

    """

    consecutive_failures: int = 0
    cooldown_until: float = 0.0


# ---------------------------------------------------------------------------
# ModelHealthTracker
# ---------------------------------------------------------------------------


class ModelHealthTracker:
    """Track model health across calls for cooldown-based fallback skipping.

    Thread-compatible (not thread-safe) — intended for use within a single
    asyncio event loop where calls are serialised.
    """

    def __init__(
        self,
        cooldown_duration: float | None = None,
        failure_threshold: int | None = None,
    ):
        """Create a tracker with optional overrides for cooldown parameters.

        Args:
            cooldown_duration: Cooldown window in seconds.  When ``None``,
                reads ``LLMIO_COOLDOWN_DURATION_SECONDS`` (default 21600).
            failure_threshold: Consecutive terminal failures needed to
                trigger cooldown.  When ``None``, reads
                ``LLMIO_COOLDOWN_FAILURE_THRESHOLD`` (default 3).

        """
        self._states: dict[str, ModelHealthState] = {}
        # family-prefix -> monotonic deadline for the shared subscription latch
        self._family_cooldown: dict[str, float] = {}
        self.cooldown_duration = (
            cooldown_duration
            if cooldown_duration is not None
            else _env_float(_ENV_COOLDOWN_DURATION, _DEFAULT_COOLDOWN_DURATION)
        )
        self.failure_threshold = (
            failure_threshold
            if failure_threshold is not None
            else _env_int(_ENV_COOLDOWN_THRESHOLD, _DEFAULT_FAILURE_THRESHOLD)
        )

    # -- public API ---------------------------------------------------------

    def is_in_cooldown(self, model_id: str, now: float | None = None) -> bool:
        """Return ``True`` if *model_id* is currently in cooldown.

        Args:
            model_id: The combined provider-model identifier (e.g.
                ``"claudeSDK-claude-fable-5"``).
            now: Monotonic timestamp for test injection; defaults to
                ``time.monotonic()``.

        """
        if now is None:
            now = time.monotonic()

        # Provider-family latch: a single exhaustion arms the whole family.
        family = _family_of(model_id)
        if family is not None:
            deadline = self._family_cooldown.get(family)
            if deadline is not None and now < deadline:
                return True

        state = self._states.get(model_id)
        if state is None:
            return False
        return now < state.cooldown_until

    def record_failure(
        self,
        model_id: str,
        now: float | None = None,
        *,
        is_terminal_fn: Callable[[BaseException], bool] = is_usage_exhausted,
        exc: BaseException | None = None,
        wall_now: datetime | None = None,
    ) -> None:
        """Record a failure for *model_id*.

        Only *terminal* failures (as determined by *is_terminal_fn*) count
        toward the cooldown threshold.  Transient failures are ignored.

        A *usage-exhaustion* failure (:func:`is_usage_exhausted`) on a
        ``claudeSDK-`` model arms the shared provider-family latch
        immediately (threshold 1) — see the module docstring. The latch
        expiry is derived from the reset hint in *exc*'s text, clamped to
        6 hours, falling back to :attr:`cooldown_duration` when unparseable.

        Args:
            model_id: The combined provider-model identifier.
            now: Monotonic timestamp for test injection.
            is_terminal_fn: Callable ``(BaseException) -> bool`` that
                returns ``True`` for terminal failures.  Defaults to
                :func:`~robotsix_llmio.core.retry.is_usage_exhausted`.
            exc: The exception that caused the failure.  When ``None``,
                the failure is always recorded (caller asserts terminal).
            wall_now: Wall-clock timestamp for test injection when deriving
                the family latch deadline from a reset hint.  Defaults to
                ``datetime.now(UTC)``.

        """
        if exc is not None and not is_terminal_fn(exc):
            return

        if now is None:
            now = time.monotonic()

        # Provider-family latch: a usage-exhaustion failure on a claudeSDK-*
        # model arms cooldown for the whole subscription immediately, so no
        # sibling tier is probed until the parsed reset time.
        family = _family_of(model_id)
        if family is not None and exc is not None and is_usage_exhausted(exc):
            if wall_now is None:
                wall_now = datetime.now(UTC)
            delay = _parse_reset_delay(str(exc), wall_now)
            if delay is None:
                delay = self.cooldown_duration
            else:
                delay = min(delay, _MAX_RESET_CLAMP_SECONDS)
            self._family_cooldown[family] = now + delay
            return

        state = self._states.setdefault(model_id, ModelHealthState())
        state.consecutive_failures += 1

        if state.consecutive_failures >= self.failure_threshold:
            state.cooldown_until = now + self.cooldown_duration

    def record_success(self, model_id: str) -> None:
        """Clear all health state for *model_id* — model is healthy again.

        A success on any ``claudeSDK-`` model also clears the shared family
        latch: if the subscription can serve this model, the quota reset has
        arrived (possibly earlier than the hinted time).
        """
        family = _family_of(model_id)
        if family is not None:
            self._family_cooldown.pop(family, None)
        self._states.pop(model_id, None)

    def reset(self) -> None:
        """Clear all tracked state (useful for tests)."""
        self._states.clear()
        self._family_cooldown.clear()


# ---------------------------------------------------------------------------
# Module-level singleton — the default tracker used by tier_fallback.
# ---------------------------------------------------------------------------

_health_tracker: ModelHealthTracker | None = None


def get_health_tracker() -> ModelHealthTracker:
    """Return the module-level singleton :class:`ModelHealthTracker`.

    Created lazily on first call with defaults from environment variables.
    """
    global _health_tracker
    if _health_tracker is None:
        _health_tracker = ModelHealthTracker()
    return _health_tracker


def reset_health_tracker() -> None:
    """Reset the module-level singleton (for test teardown)."""
    global _health_tracker
    _health_tracker = None

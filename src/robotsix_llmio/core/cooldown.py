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
"""

from __future__ import annotations

import os
import time
from collections.abc import Callable
from dataclasses import dataclass

from .retry import is_usage_exhausted

# ---------------------------------------------------------------------------
# Environment variable names (module-level constants so a rename is one line)
# ---------------------------------------------------------------------------

_ENV_COOLDOWN_DURATION = "LLMIO_COOLDOWN_DURATION_SECONDS"
_ENV_COOLDOWN_THRESHOLD = "LLMIO_COOLDOWN_FAILURE_THRESHOLD"

_DEFAULT_COOLDOWN_DURATION: float = 6 * 3600.0  # 6 hours
_DEFAULT_FAILURE_THRESHOLD: int = 3


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
    ) -> None:
        """Record a failure for *model_id*.

        Only *terminal* failures (as determined by *is_terminal_fn*) count
        toward the cooldown threshold.  Transient failures are ignored.

        Args:
            model_id: The combined provider-model identifier.
            now: Monotonic timestamp for test injection.
            is_terminal_fn: Callable ``(BaseException) -> bool`` that
                returns ``True`` for terminal failures.  Defaults to
                :func:`~robotsix_llmio.core.retry.is_usage_exhausted`.
            exc: The exception that caused the failure.  When ``None``,
                the failure is always recorded (caller asserts terminal).

        """
        if exc is not None and not is_terminal_fn(exc):
            return

        if now is None:
            now = time.monotonic()
        state = self._states.setdefault(model_id, ModelHealthState())
        state.consecutive_failures += 1

        if state.consecutive_failures >= self.failure_threshold:
            state.cooldown_until = now + self.cooldown_duration

    def record_success(self, model_id: str) -> None:
        """Clear all health state for *model_id* — model is healthy again."""
        self._states.pop(model_id, None)

    def reset(self) -> None:
        """Clear all tracked state (useful for tests)."""
        self._states.clear()


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

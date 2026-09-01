"""Unit tests for provider failover — tracker routing, arming, and status."""

from __future__ import annotations

from datetime import UTC, datetime

import pytest

from robotsix_llmio.config.tier import FailoverConfig, TierConfig
from robotsix_llmio.core.failover import (
    FailoverStatus,
    ProviderFailoverTracker,
    get_failover_status,
    get_failover_tracker,
    is_provider_shaped,
    reset_failover_tracker,
)
from robotsix_llmio.exceptions import ProviderExhaustedError


class _Exhausted(ProviderExhaustedError):
    """Provider-wide exhaustion for tests."""


def _transient() -> Exception:
    """A transient-classified failure (httpx timeout is in the signature)."""
    import httpx

    return httpx.ReadTimeout("slow upstream")


def _task_shaped() -> Exception:
    """A failure that points at the task, not the provider."""
    return ValueError("caller bug")


# --------------------------------------------------------------------------- #
#  Classification                                                             #
# --------------------------------------------------------------------------- #


def test_transient_is_provider_shaped():
    assert is_provider_shaped(_transient())


def test_exhaustion_is_provider_shaped_through_cause_chain():
    outer = RuntimeError("wrapped")
    outer.__cause__ = _Exhausted("out of credits")
    assert is_provider_shaped(outer)


def test_task_shaped_failure_is_not_provider_shaped():
    assert not is_provider_shaped(_task_shaped())


# --------------------------------------------------------------------------- #
#  Tracker routing                                                            #
# --------------------------------------------------------------------------- #


def test_starts_on_default_slot():
    tracker = ProviderFailoverTracker()
    assert tracker.active_slot() == "default"


def test_failures_below_threshold_keep_default_active():
    tracker = ProviderFailoverTracker(FailoverConfig(failure_threshold=3))
    tracker.record_failure("default", _transient(), now=0.0)
    tracker.record_failure("default", _transient(), now=1.0)
    assert tracker.active_slot(now=2.0) == "default"
    assert tracker.status(now=2.0).consecutive_failures == 2


def test_threshold_failures_arm_failover_for_window():
    tracker = ProviderFailoverTracker(
        FailoverConfig(failure_threshold=3, window_seconds=900.0)
    )
    for t in (0.0, 1.0, 2.0):
        tracker.record_failure("default", _transient(), now=t)
    assert tracker.active_slot(now=3.0) == "fallback"
    # Still armed just before expiry, back to default right after.
    assert tracker.active_slot(now=2.0 + 899.9) == "fallback"
    assert tracker.active_slot(now=2.0 + 900.1) == "default"


def test_exhaustion_arms_failover_immediately():
    tracker = ProviderFailoverTracker(
        FailoverConfig(failure_threshold=3, window_seconds=900.0)
    )
    tracker.record_failure("default", _Exhausted("resets later"), now=0.0)
    assert tracker.active_slot(now=1.0) == "fallback"


def test_dead_credential_arms_failover_immediately():
    """A ClaudeSDKAuthError (name-matched, no claude_sdk import in core)
    dooms every subscription call until the credential is refreshed —
    exactly like exhaustion, so it must not wait for the threshold."""

    class ClaudeSDKAuthError(Exception):
        pass

    tracker = ProviderFailoverTracker(FailoverConfig(failure_threshold=3))
    tracker.record_failure("default", ClaudeSDKAuthError("401"), now=0.0)
    assert tracker.active_slot(now=1.0) == "fallback"


def test_exhaustion_in_cause_chain_arms_immediately():
    outer = RuntimeError("agent run failed")
    outer.__cause__ = _Exhausted("out of credits")
    tracker = ProviderFailoverTracker(FailoverConfig(failure_threshold=3))
    tracker.record_failure("default", outer, now=0.0)
    assert tracker.active_slot(now=1.0) == "fallback"


def test_default_success_resets_streak():
    tracker = ProviderFailoverTracker(FailoverConfig(failure_threshold=3))
    tracker.record_failure("default", _transient(), now=0.0)
    tracker.record_failure("default", _transient(), now=1.0)
    tracker.record_success("default")
    tracker.record_failure("default", _transient(), now=2.0)
    # The streak restarted at zero — one failure after the success.
    assert tracker.active_slot(now=3.0) == "default"
    assert tracker.status(now=3.0).consecutive_failures == 1


def test_fallback_failures_do_not_arm_failover():
    tracker = ProviderFailoverTracker(FailoverConfig(failure_threshold=1))
    tracker.record_failure("fallback", _transient(), now=0.0)
    assert tracker.active_slot(now=1.0) == "default"
    # ...but they ARE surfaced in status for the UI.
    assert tracker.status(now=1.0).last_failure_reason is not None


def test_fallback_success_does_not_end_window_early():
    """The window alone decides when calls return to default (spec: 15 min
    on fallback, then back)."""
    tracker = ProviderFailoverTracker(
        FailoverConfig(failure_threshold=1, window_seconds=900.0)
    )
    tracker.record_failure("default", _transient(), now=0.0)
    assert tracker.active_slot(now=1.0) == "fallback"
    tracker.record_success("fallback")
    assert tracker.active_slot(now=2.0) == "fallback"


def test_default_probe_success_after_expiry_clears_state():
    tracker = ProviderFailoverTracker(
        FailoverConfig(failure_threshold=1, window_seconds=100.0)
    )
    tracker.record_failure("default", _transient(), now=0.0)
    assert tracker.active_slot(now=200.0) == "default"  # window expired
    tracker.record_success("default")
    status = tracker.status(now=201.0)
    assert not status.failover_active
    assert status.consecutive_failures == 0


def test_still_broken_default_rearms_after_expiry():
    tracker = ProviderFailoverTracker(
        FailoverConfig(failure_threshold=1, window_seconds=100.0)
    )
    tracker.record_failure("default", _transient(), now=0.0)
    assert tracker.active_slot(now=150.0) == "default"  # probe time
    tracker.record_failure("default", _transient(), now=150.0)
    assert tracker.active_slot(now=151.0) == "fallback"  # fresh window
    assert tracker.active_slot(now=249.0) == "fallback"


def test_configure_adopts_new_policy():
    tracker = ProviderFailoverTracker(FailoverConfig(failure_threshold=5))
    tracker.configure(FailoverConfig(failure_threshold=1, window_seconds=50.0))
    tracker.record_failure("default", _transient(), now=0.0)
    assert tracker.active_slot(now=1.0) == "fallback"
    assert tracker.active_slot(now=51.0) == "default"


# --------------------------------------------------------------------------- #
#  Status                                                                     #
# --------------------------------------------------------------------------- #


def test_status_snapshot_while_armed():
    tracker = ProviderFailoverTracker(
        FailoverConfig(failure_threshold=1, window_seconds=900.0)
    )
    wall = datetime(2026, 9, 1, 12, 0, tzinfo=UTC)
    tracker.record_failure("default", _Exhausted("out"), now=0.0, wall_now=wall)

    status = tracker.status(now=300.0)
    assert isinstance(status, FailoverStatus)
    assert status.active_slot == "fallback"
    assert status.failover_active
    assert status.failover_until is not None
    assert (status.failover_until - wall).total_seconds() == pytest.approx(900.0)
    assert status.seconds_remaining == pytest.approx(600.0)
    assert "out" in (status.last_failure_reason or "")
    assert status.last_failure_at == wall


def test_status_serialises_to_json():
    """Consumers ship this straight out of their status endpoints."""
    tracker = ProviderFailoverTracker()
    dumped = tracker.status().model_dump(mode="json")
    assert dumped["active_slot"] == "default"
    assert dumped["failover_active"] is False


def test_module_singleton_roundtrip():
    tracker = get_failover_tracker()
    assert get_failover_tracker() is tracker
    tracker.record_failure("default", _Exhausted("x"), now=0.0)
    assert get_failover_status().active_slot in ("default", "fallback")
    reset_failover_tracker()
    assert get_failover_tracker() is not tracker


def test_tier_config_for_level_follows_active_slot():
    """``for_level`` with no explicit slot resolves via the tracker."""
    cfg = TierConfig(failover=FailoverConfig(failure_threshold=1))
    assert cfg.for_level(2).model == "claudeSDK-opus"
    get_failover_tracker().configure(cfg.failover)
    get_failover_tracker().record_failure("default", _Exhausted("out"))
    assert cfg.for_level(2).model == "openrouter-deepseek/deepseek-v4-flash-20260731"
    assert cfg.for_level(2, slot="default").model == "claudeSDK-opus"

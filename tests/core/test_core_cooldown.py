"""Tests for model health tracking and cooldown-based fallback skipping."""

from __future__ import annotations

from unittest.mock import patch

import pytest
from pydantic_ai import UsageLimitExceeded

from robotsix_llmio.config.tier import (
    LEVEL1_DEFAULT,
    LEVEL2_DEFAULT,
    LEVEL3_DEFAULT,
    TierConfig,
    TierLevel,
)
from robotsix_llmio.core.cooldown import (
    ModelHealthTracker,
    get_health_tracker,
    reset_health_tracker,
)
from robotsix_llmio.core.tier_fallback import call_with_tier_fallback

# ---------------------------------------------------------------------------
# ModelHealthTracker unit tests
# ---------------------------------------------------------------------------


class TestModelHealthTracker:
    """Unit tests for ModelHealthTracker — cooldown entry, probe, and reset."""

    def test_starts_with_no_models_in_cooldown(self):
        tracker = ModelHealthTracker()
        assert not tracker.is_in_cooldown("test-model")

    def test_records_consecutive_failures(self):
        tracker = ModelHealthTracker(failure_threshold=3)
        exc = UsageLimitExceeded("budget exhausted")

        tracker.record_failure("test-model", now=100.0, exc=exc)
        assert not tracker.is_in_cooldown("test-model", now=100.0)

        tracker.record_failure("test-model", now=101.0, exc=exc)
        assert not tracker.is_in_cooldown("test-model", now=101.0)

        tracker.record_failure("test-model", now=102.0, exc=exc)
        # Third failure triggers cooldown
        assert tracker.is_in_cooldown("test-model", now=102.0)

    def test_enters_cooldown_at_threshold(self):
        tracker = ModelHealthTracker(cooldown_duration=600.0, failure_threshold=2)
        exc = UsageLimitExceeded("budget exhausted")

        tracker.record_failure("m", now=0.0, exc=exc)  # 1 failure
        assert not tracker.is_in_cooldown("m", now=0.0)

        tracker.record_failure("m", now=1.0, exc=exc)  # 2 failures → cooldown
        assert tracker.is_in_cooldown("m", now=1.0)
        assert tracker.is_in_cooldown("m", now=600.0)  # still in cooldown
        assert not tracker.is_in_cooldown("m", now=601.0)  # cooldown expired

    def test_ignores_transient_failures(self):
        """Non-terminal exceptions (not is_usage_exhausted) don't count."""
        tracker = ModelHealthTracker(failure_threshold=2)

        # A ValueError is not usage-exhausted
        tracker.record_failure("m", now=0.0, exc=ValueError("bad"))
        assert not tracker.is_in_cooldown("m", now=0.0)

        # Even after many transient failures, no cooldown
        for i in range(10):
            tracker.record_failure("m", now=float(i), exc=ValueError("bad"))
        assert not tracker.is_in_cooldown("m", now=10.0)

    def test_counts_terminal_failures_even_when_interleaved_with_transient(self):
        """Transient failures don't reset or affect the terminal counter."""
        tracker = ModelHealthTracker(failure_threshold=2)

        tracker.record_failure(
            "m", now=0.0, exc=UsageLimitExceeded("cap")
        )  # terminal 1
        tracker.record_failure("m", now=1.0, exc=ValueError("transient"))  # ignored
        tracker.record_failure(
            "m", now=2.0, exc=UsageLimitExceeded("cap")
        )  # terminal 2 → cooldown

        assert tracker.is_in_cooldown("m", now=2.0)

    def test_records_failure_when_exc_is_none(self):
        """When exc is None, the failure is always recorded."""
        tracker = ModelHealthTracker(failure_threshold=2)

        tracker.record_failure("m", now=0.0)  # no exc → always counted
        tracker.record_failure("m", now=1.0)  # 2nd → cooldown
        assert tracker.is_in_cooldown("m", now=1.0)

    def test_uses_custom_terminal_fn(self):
        """Custom is_terminal_fn allows different terminal classifications."""
        tracker = ModelHealthTracker(failure_threshold=2)

        def _all_terminal(_exc: BaseException) -> bool:
            return True

        tracker.record_failure(
            "m", now=0.0, is_terminal_fn=_all_terminal, exc=ValueError("any")
        )
        tracker.record_failure(
            "m", now=1.0, is_terminal_fn=_all_terminal, exc=ValueError("any")
        )
        assert tracker.is_in_cooldown("m", now=1.0)

    def test_success_clears_all_state(self):
        tracker = ModelHealthTracker(failure_threshold=2)
        tracker.record_failure("m", now=0.0, exc=UsageLimitExceeded("cap"))
        tracker.record_failure("m", now=1.0, exc=UsageLimitExceeded("cap"))
        assert tracker.is_in_cooldown("m", now=1.0)

        tracker.record_success("m")
        assert not tracker.is_in_cooldown("m", now=1.0)

    def test_probe_success_clears_cooldown(self):
        """After cooldown expires, a successful probe clears all state."""
        tracker = ModelHealthTracker(cooldown_duration=600.0, failure_threshold=2)
        tracker.record_failure("m", now=0.0, exc=UsageLimitExceeded("cap"))
        tracker.record_failure("m", now=1.0, exc=UsageLimitExceeded("cap"))
        assert tracker.is_in_cooldown("m", now=1.0)
        assert not tracker.is_in_cooldown("m", now=601.0)  # probe window

        tracker.record_success("m")
        assert not tracker.is_in_cooldown("m", now=700.0)
        # After success, a single failure shouldn't trigger cooldown
        tracker.record_failure("m", now=701.0, exc=UsageLimitExceeded("cap"))
        assert not tracker.is_in_cooldown("m", now=701.0)

    def test_probe_failure_re_arms_cooldown(self):
        """After cooldown expires, a probe failure re-arms cooldown."""
        tracker = ModelHealthTracker(cooldown_duration=600.0, failure_threshold=2)
        tracker.record_failure("m", now=0.0, exc=UsageLimitExceeded("cap"))
        tracker.record_failure("m", now=1.0, exc=UsageLimitExceeded("cap"))
        assert tracker.is_in_cooldown("m", now=1.0)
        assert not tracker.is_in_cooldown("m", now=601.0)  # probe allowed

        # Probe fails
        tracker.record_failure("m", now=601.0, exc=UsageLimitExceeded("cap"))
        # Should be back in cooldown
        assert tracker.is_in_cooldown("m", now=601.0)
        assert tracker.is_in_cooldown("m", now=1200.0)
        assert not tracker.is_in_cooldown("m", now=1201.0)  # next probe window

    def test_reset_clears_all_state(self):
        tracker = ModelHealthTracker(failure_threshold=1)
        tracker.record_failure("m1", now=0.0, exc=UsageLimitExceeded("cap"))
        tracker.record_failure("m2", now=0.0, exc=UsageLimitExceeded("cap"))
        assert tracker.is_in_cooldown("m1", now=0.0)
        assert tracker.is_in_cooldown("m2", now=0.0)

        tracker.reset()
        assert not tracker.is_in_cooldown("m1", now=0.0)
        assert not tracker.is_in_cooldown("m2", now=0.0)

    def test_tracks_multiple_models_independently(self):
        tracker = ModelHealthTracker(failure_threshold=2)
        tracker.record_failure("m1", now=0.0, exc=UsageLimitExceeded("cap"))
        tracker.record_failure("m1", now=1.0, exc=UsageLimitExceeded("cap"))
        assert tracker.is_in_cooldown("m1", now=1.0)

        tracker.record_failure("m2", now=0.0, exc=UsageLimitExceeded("cap"))
        assert not tracker.is_in_cooldown("m2", now=0.0)


# ---------------------------------------------------------------------------
# Singleton tests
# ---------------------------------------------------------------------------


class TestSingletonHealthTracker:
    """Tests for module-level singleton — get_health_tracker / reset."""

    def test_returns_same_instance_on_repeated_calls(self):
        reset_health_tracker()
        t1 = get_health_tracker()
        t2 = get_health_tracker()
        assert t1 is t2

    def test_reset_creates_new_instance(self):
        reset_health_tracker()
        t1 = get_health_tracker()
        reset_health_tracker()
        t2 = get_health_tracker()
        assert t1 is not t2


# ---------------------------------------------------------------------------
# Integration tests: tier_fallback with cooldown
# ---------------------------------------------------------------------------


class TestTierFallbackWithCooldown:
    """Integration tests: tier_fallback loop respects cooldown state."""

    @pytest.fixture(autouse=True)
    def _reset_tracker(self):
        """Ensure a clean tracker for every test."""
        reset_health_tracker()

    def _tier_config(self, level1_model: str = "claudeSDK-level1-test") -> TierConfig:
        """Build a TierConfig with custom models for isolation."""
        return TierConfig(
            level1=LEVEL1_DEFAULT.model_copy(update={"model": level1_model}),
            level2=LEVEL2_DEFAULT.model_copy(update={"model": "claudeSDK-level2-test"}),
            level3=LEVEL3_DEFAULT.model_copy(update={"model": "claudeSDK-level3-test"}),
        )

    def test_skips_model_in_cooldown(self):
        """When level1 model is in cooldown, tier fallback skips to level2."""
        tracker = get_health_tracker()
        # Manually put level1 in cooldown — use real monotonic time so
        # cooldown_until is in the future.
        for _ in range(tracker.failure_threshold):
            tracker.record_failure(
                "claudeSDK-level1-test", exc=UsageLimitExceeded("cap")
            )

        cfg = self._tier_config(level1_model="claudeSDK-level1-test")
        call_order: list[str] = []

        def factory(tlc):
            model = tlc.model

            def call():
                call_order.append(model)
                return f"result-{model}"

            return call

        result = call_with_tier_fallback(
            factory,
            tier_config=cfg,
            level=TierLevel.LEVEL1,
            fallback_enabled=True,
            max_fallback_depth=2,
        )
        # level1 is in cooldown → skipped, level2 tried and succeeds
        assert result == "result-claudeSDK-level2-test"
        assert "claudeSDK-level1-test" not in call_order
        assert "claudeSDK-level2-test" in call_order

    def test_records_terminal_failure_and_skips_on_next_call(self):
        """After a terminal failure, subsequent calls skip the model."""
        tracker = get_health_tracker()
        # Override threshold to 1 so a single failure triggers cooldown
        tracker.failure_threshold = 1

        cfg = self._tier_config(level1_model="claudeSDK-level1-test")

        # First call: level1 fails terminally → fallback to level2
        call_count = 0

        def factory_first(tlc):
            nonlocal call_count

            def call():
                nonlocal call_count
                call_count += 1
                if tlc.model == "claudeSDK-level1-test":
                    raise UsageLimitExceeded("cap exceeded")
                return "ok"

            return call

        result = call_with_tier_fallback(
            factory_first,
            tier_config=cfg,
            level=TierLevel.LEVEL1,
            fallback_enabled=True,
        )
        assert result == "ok"
        assert call_count == 2  # level1 failed, level2 succeeded

        # Second call: level1 is now in cooldown → skipped entirely
        call_count = 0

        def factory_second(tlc):
            nonlocal call_count

            def call():
                nonlocal call_count
                call_count += 1
                return f"result-{tlc.model}"

            return call

        result = call_with_tier_fallback(
            factory_second,
            tier_config=cfg,
            level=TierLevel.LEVEL1,
            fallback_enabled=True,
        )
        assert result == "result-claudeSDK-level2-test"
        assert call_count == 1  # only level2 was called

    def test_clears_cooldown_on_success(self):
        """After a success, the cooldown state for that model is cleared."""
        tracker = get_health_tracker()
        tracker.failure_threshold = 1

        cfg = self._tier_config(level1_model="claudeSDK-level1-test")

        # First call: level1 fails → cooldown
        def factory_fail(tlc):
            def call():
                if tlc.model == "claudeSDK-level1-test":
                    raise UsageLimitExceeded("cap")
                return "ok"

            return call

        call_with_tier_fallback(
            factory_fail,
            tier_config=cfg,
            level=TierLevel.LEVEL1,
            fallback_enabled=True,
        )
        # Model is in cooldown (no now= arg — uses real monotonic time)
        assert tracker.is_in_cooldown("claudeSDK-level1-test")

        # Simulate cooldown expiry by resetting the tracker
        tracker.reset()

        # Second call: level1 succeeds → cooldown cleared
        def factory_ok(tlc):
            def call():
                return f"result-{tlc.model}"

            return call

        call_with_tier_fallback(
            factory_ok,
            tier_config=cfg,
            level=TierLevel.LEVEL1,
            fallback_enabled=True,
        )
        assert not tracker.is_in_cooldown("claudeSDK-level1-test")

    @patch("robotsix_llmio.core.cooldown.time.monotonic")
    def test_allows_probe_after_cooldown_expiry(self, mock_monotonic):
        """After cooldown expires, the model is attempted (probe)."""
        tracker = get_health_tracker()
        tracker.failure_threshold = 1
        tracker.cooldown_duration = 600.0

        cfg = self._tier_config(level1_model="claudeSDK-level1-test")

        # Set time and make level1 fail
        mock_monotonic.return_value = 100.0

        def factory_fail(tlc):
            def call():
                if tlc.model == "claudeSDK-level1-test":
                    raise UsageLimitExceeded("cap")
                return "ok"

            return call

        call_with_tier_fallback(
            factory_fail,
            tier_config=cfg,
            level=TierLevel.LEVEL1,
            fallback_enabled=True,
        )
        assert tracker.is_in_cooldown("claudeSDK-level1-test", now=100.0)
        assert not tracker.is_in_cooldown("claudeSDK-level1-test", now=701.0)

        # Advance time past cooldown
        mock_monotonic.return_value = 701.0

        call_order: list[str] = []

        def factory_probe(tlc):
            def call():
                call_order.append(tlc.model)
                return f"result-{tlc.model}"

            return call

        result = call_with_tier_fallback(
            factory_probe,
            tier_config=cfg,
            level=TierLevel.LEVEL1,
            fallback_enabled=True,
        )
        # level1 is probed and succeeds
        assert result == "result-claudeSDK-level1-test"
        assert "claudeSDK-level1-test" in call_order

    def test_raises_when_all_tiers_in_cooldown(self):
        """When all available tiers are in cooldown, a RuntimeError is raised."""
        tracker = get_health_tracker()
        tracker.failure_threshold = 1

        cfg = self._tier_config(level1_model="claudeSDK-level1-test")

        # Put all three tiers in cooldown.
        tracker.record_failure("claudeSDK-level1-test", exc=UsageLimitExceeded("cap"))
        tracker.record_failure("claudeSDK-level2-test", exc=UsageLimitExceeded("cap"))
        tracker.record_failure("claudeSDK-level3-test", exc=UsageLimitExceeded("cap"))

        def factory(tlc):
            def call():
                return "ok"

            return call

        with pytest.raises(RuntimeError, match="cooldown"):
            call_with_tier_fallback(
                factory,
                tier_config=cfg,
                level=TierLevel.LEVEL1,
                fallback_enabled=True,
            )

    def test_raises_when_cooldown_and_fallback_depth_exhausted(self):
        """When a tier is in cooldown and fallback depth is 0, raise."""
        tracker = get_health_tracker()
        tracker.failure_threshold = 1

        cfg = self._tier_config(level1_model="claudeSDK-level1-test")
        tracker.record_failure("claudeSDK-level1-test", exc=UsageLimitExceeded("cap"))

        def factory(tlc):
            def call():
                return "ok"

            return call

        with pytest.raises(RuntimeError, match="cooldown"):
            call_with_tier_fallback(
                factory,
                tier_config=cfg,
                level=TierLevel.LEVEL1,
                fallback_enabled=True,
                max_fallback_depth=0,
            )

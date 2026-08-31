"""Tests for model health tracking and cooldown-based fallback skipping."""

from __future__ import annotations

from datetime import UTC, datetime
from unittest.mock import patch

import pytest
from pydantic_ai import UsageLimitExceeded

from robotsix_llmio.claude_sdk._errors import ClaudeSDKUsageExhaustedError
from robotsix_llmio.config.tier import (
    LEVEL1_DEFAULT,
    LEVEL2_DEFAULT,
    LEVEL3_DEFAULT,
    TierConfig,
    TierLevel,
)
from robotsix_llmio.core.cooldown import (
    ModelHealthTracker,
    _parse_reset_delay,
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

    def _tier_config(self, level1_model: str = "openrouter-level1-test") -> TierConfig:
        """Build a TierConfig with custom models for isolation."""
        return TierConfig(
            level1=LEVEL1_DEFAULT.model_copy(update={"model": level1_model}),
            level2=LEVEL2_DEFAULT.model_copy(
                update={"model": "openrouter-level2-test"}
            ),
            level3=LEVEL3_DEFAULT.model_copy(
                update={"model": "openrouter-level3-test"}
            ),
        )

    def test_skips_model_in_cooldown(self):
        """When level1 model is in cooldown, tier fallback skips to level2."""
        tracker = get_health_tracker()
        # Manually put level1 in cooldown — use real monotonic time so
        # cooldown_until is in the future.
        for _ in range(tracker.failure_threshold):
            tracker.record_failure(
                "openrouter-level1-test", exc=UsageLimitExceeded("cap")
            )

        cfg = self._tier_config(level1_model="openrouter-level1-test")
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
        assert result == "result-openrouter-level2-test"
        assert "openrouter-level1-test" not in call_order
        assert "openrouter-level2-test" in call_order

    def test_records_terminal_failure_and_skips_on_next_call(self):
        """After a terminal failure, subsequent calls skip the model."""
        tracker = get_health_tracker()
        # Override threshold to 1 so a single failure triggers cooldown
        tracker.failure_threshold = 1

        cfg = self._tier_config(level1_model="openrouter-level1-test")

        # First call: level1 fails terminally → fallback to level2
        call_count = 0

        def factory_first(tlc):
            nonlocal call_count

            def call():
                nonlocal call_count
                call_count += 1
                if tlc.model == "openrouter-level1-test":
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
        assert result == "result-openrouter-level2-test"
        assert call_count == 1  # only level2 was called

    def test_clears_cooldown_on_success(self):
        """After a success, the cooldown state for that model is cleared."""
        tracker = get_health_tracker()
        tracker.failure_threshold = 1

        cfg = self._tier_config(level1_model="openrouter-level1-test")

        # First call: level1 fails → cooldown
        def factory_fail(tlc):
            def call():
                if tlc.model == "openrouter-level1-test":
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
        assert tracker.is_in_cooldown("openrouter-level1-test")

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
        assert not tracker.is_in_cooldown("openrouter-level1-test")

    @patch("robotsix_llmio.core.cooldown.time.monotonic")
    def test_allows_probe_after_cooldown_expiry(self, mock_monotonic):
        """After cooldown expires, the model is attempted (probe)."""
        tracker = get_health_tracker()
        tracker.failure_threshold = 1
        tracker.cooldown_duration = 600.0

        cfg = self._tier_config(level1_model="openrouter-level1-test")

        # Set time and make level1 fail
        mock_monotonic.return_value = 100.0

        def factory_fail(tlc):
            def call():
                if tlc.model == "openrouter-level1-test":
                    raise UsageLimitExceeded("cap")
                return "ok"

            return call

        call_with_tier_fallback(
            factory_fail,
            tier_config=cfg,
            level=TierLevel.LEVEL1,
            fallback_enabled=True,
        )
        assert tracker.is_in_cooldown("openrouter-level1-test", now=100.0)
        assert not tracker.is_in_cooldown("openrouter-level1-test", now=701.0)

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
        assert result == "result-openrouter-level1-test"
        assert "openrouter-level1-test" in call_order

    def test_raises_when_all_tiers_in_cooldown(self):
        """When all available tiers are in cooldown, a RuntimeError is raised."""
        tracker = get_health_tracker()
        tracker.failure_threshold = 1

        cfg = self._tier_config(level1_model="openrouter-level1-test")

        # Put all three tiers in cooldown.
        tracker.record_failure("openrouter-level1-test", exc=UsageLimitExceeded("cap"))
        tracker.record_failure("openrouter-level2-test", exc=UsageLimitExceeded("cap"))
        tracker.record_failure("openrouter-level3-test", exc=UsageLimitExceeded("cap"))

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

        cfg = self._tier_config(level1_model="openrouter-level1-test")
        tracker.record_failure("openrouter-level1-test", exc=UsageLimitExceeded("cap"))

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


# ---------------------------------------------------------------------------
# Reset-hint parsing
# ---------------------------------------------------------------------------


class TestParseResetDelay:
    """Unit tests for ``_parse_reset_delay`` — the ``resets H[:MM]am|pm (UTC)``
    hint the Claude CLI carries on exhaustion."""

    def test_pm_with_minutes(self):
        wall = datetime(2026, 1, 1, 12, 0, tzinfo=UTC)
        # 1:10pm -> 13:10, 70 minutes ahead
        assert _parse_reset_delay("resets 1:10pm (UTC)", wall) == 70 * 60

    def test_am_with_minutes(self):
        wall = datetime(2026, 1, 1, 10, 0, tzinfo=UTC)
        # 11:10am -> 11:10, 70 minutes ahead
        assert _parse_reset_delay("resets 11:10am (UTC)", wall) == 70 * 60

    def test_pm_without_minutes(self):
        wall = datetime(2026, 1, 1, 12, 0, tzinfo=UTC)
        # 8pm -> 20:00, 8 hours ahead
        assert _parse_reset_delay("resets 8pm (UTC)", wall) == 8 * 3600

    def test_am_without_minutes(self):
        wall = datetime(2026, 1, 1, 6, 0, tzinfo=UTC)
        # 9am -> 09:00, 3 hours ahead
        assert _parse_reset_delay("resets 9am (UTC)", wall) == 3 * 3600

    def test_rollover_to_next_day(self):
        wall = datetime(2026, 1, 1, 14, 0, tzinfo=UTC)
        # 1pm already passed today -> next day 13:00 = 23 hours ahead
        assert _parse_reset_delay("resets 1pm (UTC)", wall) == 23 * 3600

    def test_noon_is_twelve_pm(self):
        wall = datetime(2026, 1, 1, 6, 0, tzinfo=UTC)
        # 12pm -> noon, 6 hours ahead
        assert _parse_reset_delay("resets 12pm (UTC)", wall) == 6 * 3600

    def test_midnight_is_twelve_am_rolls_over(self):
        wall = datetime(2026, 1, 1, 6, 0, tzinfo=UTC)
        # 12am -> midnight already passed -> next day 00:00 = 18 hours ahead
        assert _parse_reset_delay("resets 12am (UTC)", wall) == 18 * 3600

    def test_case_insensitive_and_embedded_in_message(self):
        wall = datetime(2026, 1, 1, 12, 0, tzinfo=UTC)
        text = "You've hit your session limit · RESETS 1:00PM (utc). Try later."
        assert _parse_reset_delay(text, wall) == 3600

    def test_no_hint_returns_none(self):
        wall = datetime(2026, 1, 1, 12, 0, tzinfo=UTC)
        assert _parse_reset_delay("out of usage credits", wall) is None

    def test_invalid_hour_returns_none(self):
        wall = datetime(2026, 1, 1, 12, 0, tzinfo=UTC)
        assert _parse_reset_delay("resets 13pm (UTC)", wall) is None


# ---------------------------------------------------------------------------
# Claude SDK provider-family latch
# ---------------------------------------------------------------------------


class TestClaudeSDKFamilyLatch:
    """Unit tests for the shared ``claudeSDK-*`` subscription latch."""

    def test_first_exhaustion_arms_whole_family_at_threshold_one(self):
        """A single exhaustion arms cooldown despite a threshold of 3."""
        tracker = ModelHealthTracker(cooldown_duration=600.0, failure_threshold=3)
        exc = ClaudeSDKUsageExhaustedError("out of usage credits")
        tracker.record_failure("claudeSDK-haiku", now=0.0, exc=exc)
        assert tracker.is_in_cooldown("claudeSDK-haiku", now=0.0)

    def test_arming_one_model_skips_all_family_models(self):
        """Sibling tiers on the same subscription are skipped too."""
        tracker = ModelHealthTracker(cooldown_duration=600.0, failure_threshold=3)
        exc = ClaudeSDKUsageExhaustedError("out of usage credits")
        tracker.record_failure("claudeSDK-opus", now=0.0, exc=exc)
        assert tracker.is_in_cooldown("claudeSDK-haiku", now=0.0)
        assert tracker.is_in_cooldown("claudeSDK-claude-fable-5", now=0.0)

    def test_openrouter_models_unaffected(self):
        """Models outside the family use the ordinary per-model threshold."""
        tracker = ModelHealthTracker(cooldown_duration=600.0, failure_threshold=3)
        exc = ClaudeSDKUsageExhaustedError("out of usage credits")
        tracker.record_failure("claudeSDK-opus", now=0.0, exc=exc)
        assert not tracker.is_in_cooldown(
            "openrouter-deepseek/deepseek-v4-flash-latest", now=0.0
        )

    def test_success_on_any_family_model_clears_latch(self):
        """A success means the quota reset arrived — clear the whole family."""
        tracker = ModelHealthTracker(cooldown_duration=600.0, failure_threshold=3)
        exc = ClaudeSDKUsageExhaustedError("out of usage credits")
        tracker.record_failure("claudeSDK-opus", now=0.0, exc=exc)
        assert tracker.is_in_cooldown("claudeSDK-haiku", now=0.0)

        tracker.record_success("claudeSDK-haiku")
        assert not tracker.is_in_cooldown("claudeSDK-opus", now=0.0)

    def test_latch_uses_parsed_reset_deadline(self):
        wall = datetime(2026, 1, 1, 12, 0, tzinfo=UTC)
        tracker = ModelHealthTracker(cooldown_duration=600.0, failure_threshold=1)
        exc = ClaudeSDKUsageExhaustedError(
            "You've hit your session limit · resets 1:00pm (UTC)"
        )
        tracker.record_failure("claudeSDK-opus", now=100.0, exc=exc, wall_now=wall)
        # 1pm - 12pm = 3600s; monotonic deadline = 100 + 3600
        assert tracker.is_in_cooldown("claudeSDK-opus", now=3699.0)
        assert not tracker.is_in_cooldown("claudeSDK-opus", now=3701.0)

    def test_unparseable_reset_falls_back_to_fixed_duration(self):
        tracker = ModelHealthTracker(cooldown_duration=600.0, failure_threshold=1)
        exc = ClaudeSDKUsageExhaustedError("out of usage credits")
        tracker.record_failure("claudeSDK-opus", now=0.0, exc=exc)
        assert tracker.is_in_cooldown("claudeSDK-opus", now=599.0)
        assert not tracker.is_in_cooldown("claudeSDK-opus", now=601.0)

    def test_reset_deadline_clamped_to_six_hours(self):
        wall = datetime(2026, 1, 1, 1, 0, tzinfo=UTC)
        tracker = ModelHealthTracker(cooldown_duration=600.0, failure_threshold=1)
        # "resets 12am (UTC)" from 1am is ~23h away — clamp to the 6h max.
        exc = ClaudeSDKUsageExhaustedError("resets 12am (UTC)")
        tracker.record_failure("claudeSDK-opus", now=0.0, exc=exc, wall_now=wall)
        assert tracker.is_in_cooldown("claudeSDK-opus", now=21599.0)
        assert not tracker.is_in_cooldown("claudeSDK-opus", now=21601.0)

    def test_non_exhaustion_failure_does_not_arm_family(self):
        """A non-usage-exhausted failure uses per-model tracking, not the latch."""
        tracker = ModelHealthTracker(cooldown_duration=600.0, failure_threshold=2)
        tracker.record_failure("claudeSDK-opus", now=0.0, exc=ValueError("boom"))
        assert not tracker.is_in_cooldown("claudeSDK-opus", now=0.0)
        assert not tracker.is_in_cooldown("claudeSDK-haiku", now=0.0)


class TestFamilyLatchTierFallback:
    """Acceptance: a fallback walk makes ZERO claudeSDK calls after the first
    exhaustion, and consumers need no code change."""

    @pytest.fixture(autouse=True)
    def _reset_tracker(self):
        reset_health_tracker()

    def _mixed_config(self) -> TierConfig:
        # level1/level2 on the shared Claude subscription; level3 elsewhere.
        return TierConfig(
            level1=LEVEL1_DEFAULT.model_copy(update={"model": "claudeSDK-l1"}),
            level2=LEVEL2_DEFAULT.model_copy(update={"model": "claudeSDK-l2"}),
            level3=LEVEL3_DEFAULT.model_copy(update={"model": "openrouter-l3"}),
        )

    def test_zero_claudesdk_calls_after_first_failure(self):
        cfg = self._mixed_config()
        call_order: list[str] = []

        def factory(tlc):
            def call():
                call_order.append(tlc.model)
                if tlc.model.startswith("claudeSDK-"):
                    raise UsageLimitExceeded(
                        "You've hit your session limit · resets 8pm (UTC)"
                    )
                return f"result-{tlc.model}"

            return call

        # First walk: level1 fails, level2 (same family) is NEVER probed.
        result = call_with_tier_fallback(
            factory,
            tier_config=cfg,
            level=TierLevel.LEVEL1,
            fallback_enabled=True,
            max_fallback_depth=2,
        )
        assert result == "result-openrouter-l3"
        assert call_order.count("claudeSDK-l1") == 1
        assert "claudeSDK-l2" not in call_order

        # Second walk: ZERO claudeSDK calls at all — the family stays latched.
        call_order.clear()
        result2 = call_with_tier_fallback(
            factory,
            tier_config=cfg,
            level=TierLevel.LEVEL1,
            fallback_enabled=True,
            max_fallback_depth=2,
        )
        assert result2 == "result-openrouter-l3"
        assert not any(m.startswith("claudeSDK-") for m in call_order)

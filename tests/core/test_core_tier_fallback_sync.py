"""Sync ``call_with_tier_fallback`` tests.

Extracted from ``test_core_tier_fallback.py`` (#20260727T132934Z).
"""

from __future__ import annotations

import asyncio
import logging

import pytest
from conftest import (
    STD_TIER_CONFIG,
    _noop_sleep,
    tf_exhausted_failing_factory,
    tf_factory_that_succeeds,
)

from robotsix_llmio.config.tier import TierLevel, TierLevelConfig
from robotsix_llmio.core.tier_fallback import call_with_tier_fallback

# --------------------------------------------------------------------------- #
#  call_with_tier_fallback                                                    #
# --------------------------------------------------------------------------- #


def test_successful_call_returns_result():
    out = call_with_tier_fallback(
        tf_factory_that_succeeds("hello"),
        tier_config=STD_TIER_CONFIG,
        sleep=_noop_sleep,
    )
    assert out == "hello"


def test_fallback_disabled_raises_immediately():
    with pytest.raises(RuntimeError, match="boom"):
        call_with_tier_fallback(
            tf_exhausted_failing_factory(RuntimeError),
            tier_config=STD_TIER_CONFIG,
            fallback_enabled=False,
            sleep=_noop_sleep,
        )


def test_fallback_level1_to_level2_on_failure():
    """level1 fails once → level2 succeeds."""
    tracking: dict = {}

    def factory(tlc: TierLevelConfig):
        tracking.setdefault("factory_calls", []).append(tlc.model_name)
        if tlc.model_name == "opus":

            def fn():
                raise RuntimeError("l1-fail")
        else:

            def fn():
                return "l2-ok"

        return fn

    out = call_with_tier_fallback(
        factory,
        tier_config=STD_TIER_CONFIG,
        fallback_enabled=True,
        max_fallback_depth=2,
        sleep=_noop_sleep,
    )
    assert out == "l2-ok"
    assert tracking["factory_calls"] == ["opus", "haiku"]


def test_fallback_level1_to_level2_to_level3():
    """level1 fails, level2 fails, level3 succeeds."""
    tracking: dict = {}
    counter = {"remaining": 2}  # first 2 calls fail

    def factory(tlc: TierLevelConfig):
        tracking.setdefault("factory_calls", []).append(tlc.model_name)

        def fn():
            if counter["remaining"] > 0:
                counter["remaining"] -= 1
                raise RuntimeError(f"fail-{tlc.model_name}")
            return "l3-ok"

        return fn

    out = call_with_tier_fallback(
        factory,
        tier_config=STD_TIER_CONFIG,
        fallback_enabled=True,
        max_fallback_depth=2,
        sleep=_noop_sleep,
    )
    assert out == "l3-ok"
    assert tracking["factory_calls"] == ["opus", "haiku", "sonnet"]


def test_fallback_level2_to_level3_to_level4():
    """Start at level2: level2 fails → level3 fails → level4 succeeds.
    Escalation prefers higher tiers while any remain unvisited."""
    tracking: dict = {}
    counter = {"remaining": 2}

    def factory(tlc: TierLevelConfig):
        tracking.setdefault("factory_calls", []).append(tlc.model_name)

        def fn():
            if counter["remaining"] > 0:
                counter["remaining"] -= 1
                raise RuntimeError(f"fail-{tlc.model_name}")
            return "l4-ok"

        return fn

    out = call_with_tier_fallback(
        factory,
        tier_config=STD_TIER_CONFIG,
        level=TierLevel.LEVEL2,
        fallback_enabled=True,
        max_fallback_depth=2,
        sleep=_noop_sleep,
    )
    assert out == "l4-ok"
    assert tracking["factory_calls"] == ["haiku", "sonnet", "claude-fable-5"]


def test_fallback_level3_to_level4_to_level2():
    """Start at level3: level3 fails → level4 (higher) fails → level2
    (nearest lower) succeeds. Demonstrates bidirectional fallback."""
    tracking: dict = {}
    counter = {"remaining": 2}

    def factory(tlc: TierLevelConfig):
        tracking.setdefault("factory_calls", []).append(tlc.model_name)

        def fn():
            if counter["remaining"] > 0:
                counter["remaining"] -= 1
                raise RuntimeError(f"fail-{tlc.model_name}")
            return "l2-ok"

        return fn

    out = call_with_tier_fallback(
        factory,
        tier_config=STD_TIER_CONFIG,
        level=TierLevel.LEVEL3,
        fallback_enabled=True,
        max_fallback_depth=2,
        sleep=_noop_sleep,
    )
    assert out == "l2-ok"
    assert tracking["factory_calls"] == ["sonnet", "claude-fable-5", "haiku"]


def test_fallback_level4_to_level3_to_level2():
    """Start at level4: no higher tier exists, so fallback walks down —
    level4 fails → level3 fails → level2 succeeds."""
    tracking: dict = {}
    counter = {"remaining": 2}

    def factory(tlc: TierLevelConfig):
        tracking.setdefault("factory_calls", []).append(tlc.model_name)

        def fn():
            if counter["remaining"] > 0:
                counter["remaining"] -= 1
                raise RuntimeError(f"fail-{tlc.model_name}")
            return "l2-ok"

        return fn

    out = call_with_tier_fallback(
        factory,
        tier_config=STD_TIER_CONFIG,
        level=TierLevel.LEVEL4,
        fallback_enabled=True,
        max_fallback_depth=2,
        sleep=_noop_sleep,
    )
    assert out == "l2-ok"
    assert tracking["factory_calls"] == ["claude-fable-5", "sonnet", "haiku"]


def test_exhausted_all_levels_reraises_last_error():
    """All three levels fail → last exception re-raised."""
    with pytest.raises(RuntimeError, match="boom"):
        call_with_tier_fallback(
            tf_exhausted_failing_factory(RuntimeError),
            tier_config=STD_TIER_CONFIG,
            level=TierLevel.LEVEL1,
            fallback_enabled=True,
            max_fallback_depth=2,
            sleep=_noop_sleep,
        )


def test_factory_called_fresh_per_level():
    """Each tier gets its own TierLevelConfig via a fresh factory call."""
    tracking: dict = {}
    counter = {"remaining": 2}

    def factory(tlc: TierLevelConfig):
        tracking.setdefault("factory_calls", []).append(
            {"provider": tlc.provider, "model": tlc.model_name}
        )

        def fn():
            if counter["remaining"] > 0:
                counter["remaining"] -= 1
                raise RuntimeError("fail")
            return "ok"

        return fn

    call_with_tier_fallback(
        factory,
        tier_config=STD_TIER_CONFIG,
        fallback_enabled=True,
        max_fallback_depth=2,
        sleep=_noop_sleep,
    )

    assert tracking["factory_calls"] == [
        {"provider": "claudeSDK", "model": "opus"},
        {"provider": "claudeSDK", "model": "haiku"},
        {"provider": "claudeSDK", "model": "sonnet"},
    ]


def test_max_fallback_depth_zero_equals_disabled():
    """max_fallback_depth=0: no escalation even when fallback_enabled=True."""
    tracking: dict = {}

    def factory(tlc: TierLevelConfig):
        tracking.setdefault("factory_calls", []).append(tlc.model_name)

        def fn():
            raise RuntimeError("fail")

        return fn

    with pytest.raises(RuntimeError, match="fail"):
        call_with_tier_fallback(
            factory,
            tier_config=STD_TIER_CONFIG,
            fallback_enabled=True,
            max_fallback_depth=0,
            sleep=_noop_sleep,
        )
    assert tracking["factory_calls"] == ["opus"]


def test_max_fallback_depth_limits_promotions():
    """max_fallback_depth=1, level=LEVEL1: level1 fails → level2 tried →
    level2 fails → re-raised (level3 never reached)."""
    tracking: dict = {}

    def factory(tlc: TierLevelConfig):
        tracking.setdefault("factory_calls", []).append(tlc.model_name)

        def fn():
            raise RuntimeError(f"fail-{tlc.model_name}")

        return fn

    with pytest.raises(RuntimeError, match="fail-haiku"):
        call_with_tier_fallback(
            factory,
            tier_config=STD_TIER_CONFIG,
            level=TierLevel.LEVEL1,
            fallback_enabled=True,
            max_fallback_depth=1,
            sleep=_noop_sleep,
        )
    assert tracking["factory_calls"] == ["opus", "haiku"]


def test_no_duplicate_tier_visits():
    """Verify no tier is visited more than once.
    The visited-set prevents ping-pong: when all tiers except the starting
    one are already visited, the loop terminates instead of revisiting."""
    # Start at LEVEL2, only LEVEL2 and LEVEL3 in config, but level1 also
    # exists. The spec says no revisits. Let's construct a scenario where
    # the _next_unvisited_tier returns the correct chain without duplicates.
    # Actually the visited set naturally prevents duplicates; the test just
    # confirms that a tier succeeding on its first visit is only called once.

    tracking: dict = {}

    def factory(tlc: TierLevelConfig):
        tracking.setdefault("factory_calls", []).append(tlc.model_name)

        def fn():
            return "ok"

        return fn

    call_with_tier_fallback(
        factory,
        tier_config=STD_TIER_CONFIG,
        level=TierLevel.LEVEL2,
        fallback_enabled=True,
        sleep=_noop_sleep,
    )

    # Only one visit — no duplicates
    assert tracking["factory_calls"] == ["haiku"]

    # Also verify that when all tiers are exhausted, the loop stops and doesn't
    # revisit. Using the exhausted factory: all 3 tiers fail once each.
    tracking2: dict = {}

    def failing_factory(tlc: TierLevelConfig):
        tracking2.setdefault("factory_calls", []).append(tlc.model_name)

        def fn():
            raise RuntimeError("fail")

        return fn

    with pytest.raises(RuntimeError):
        call_with_tier_fallback(
            failing_factory,
            tier_config=STD_TIER_CONFIG,
            fallback_enabled=True,
            max_fallback_depth=2,
            sleep=_noop_sleep,
        )

    assert tracking2["factory_calls"] == ["opus", "haiku", "sonnet"]
    # No duplicates
    assert len(tracking2["factory_calls"]) == len(set(tracking2["factory_calls"]))


def test_logging_output(caplog):
    """Verify INFO log on tier attempt (with provider/model) and WARNING log
    on fallback (which tier failed, exception type, which tier promoted to)."""
    caplog.set_level(logging.INFO, logger="robotsix_llmio.tier_fallback")

    counter = {"remaining": 1}

    def factory(tlc: TierLevelConfig):
        def fn():
            if counter["remaining"] > 0:
                counter["remaining"] -= 1
                raise RuntimeError("test-fail")
            return "ok"

        return fn

    call_with_tier_fallback(
        factory,
        tier_config=STD_TIER_CONFIG,
        fallback_enabled=True,
        max_fallback_depth=2,
        what="test-op",
        sleep=_noop_sleep,
    )

    # Check INFO messages
    info_messages = [r.message for r in caplog.records if r.levelno == logging.INFO]
    assert any(
        "test-op: trying level1 (provider=claudeSDK, model=opus)" in msg
        for msg in info_messages
    )
    assert any(
        "test-op: trying level2 (provider=claudeSDK, model=haiku)" in msg
        for msg in info_messages
    )
    assert any("test-op: level2 succeeded" in msg for msg in info_messages)

    # Check WARNING messages
    warn_messages = [r.message for r in caplog.records if r.levelno == logging.WARNING]
    assert any(
        "test-op: level1 failed with RuntimeError — falling back to level2" in msg
        for msg in warn_messages
    )


def test_logging_on_success_no_warning(caplog):
    """No WARNING emitted on success path."""
    caplog.set_level(logging.INFO, logger="robotsix_llmio.tier_fallback")

    call_with_tier_fallback(
        tf_factory_that_succeeds("ok"),
        tier_config=STD_TIER_CONFIG,
        what="happy-path",
        sleep=_noop_sleep,
    )

    warn_messages = [r.message for r in caplog.records if r.levelno == logging.WARNING]
    assert len(warn_messages) == 0


def test_call_with_tier_fallback_supports_run_sync_style_fn():
    """The sync wrapper must execute the tier callable loop-free so
    run_sync-style callables (asyncio.run inside) work."""

    async def payload():
        return "ok"

    def factory(_cfg: TierLevelConfig):
        return lambda: asyncio.run(payload())

    out = call_with_tier_fallback(
        factory, tier_config=STD_TIER_CONFIG, sleep=_noop_sleep
    )
    assert out == "ok"

"""Tier-fallback escalation loop — pure-function injection tests.

Follows the existing ``test_core_retry.py`` conventions: no mocks,
pure-function injection, ``_noop_sleep``, ``asyncio.run()`` for async,
standalone ``def test_*`` functions.
"""

from __future__ import annotations

import asyncio
import logging

import pytest

from robotsix_llmio.config.tier import TierConfig, TierLevel, TierLevelConfig
from robotsix_llmio.core.tier_fallback import (
    _next_unvisited_tier,
    acall_with_tier_fallback,
    call_with_tier_fallback,
)


def _noop_sleep(_d: float) -> None:
    return None


async def _anoop_sleep(_d: float) -> None:
    return None


# --------------------------------------------------------------------------- #
#  Helpers for constructing test configs                                      #
# --------------------------------------------------------------------------- #

_L1_CFG = TierLevelConfig(model="claudeSDK-opus")
_L2_CFG = TierLevelConfig(model="claudeSDK-haiku")
_L3_CFG = TierLevelConfig(model="claudeSDK-sonnet")

_STD_TIER_CONFIG = TierConfig(level1=_L1_CFG, level2=_L2_CFG, level3=_L3_CFG)


def _factory_that_succeeds(
    result: str = "ok",
    *,
    tracking: dict | None = None,
    expected_tier: str | None = None,
):
    """Return a factory that returns a callable that succeeds."""

    def factory(tlc: TierLevelConfig):
        if tracking is not None:
            tracking.setdefault("factory_calls", []).append(tlc.model_name)
        if expected_tier is not None:
            assert tlc.model_name == expected_tier, (
                f"expected {expected_tier}, got {tlc.model_name}"
            )

        def fn():
            return result

        return fn

    return factory


def _failing_factory(
    exception: type[Exception] | Exception,
    *,
    fail_count: int = 1,
    tracking: dict | None = None,
):
    """Return a factory whose callable raises *exception* the first *fail_count*
    times it is invoked (global counter across all factory calls)."""
    counter = {"remaining": fail_count}

    def factory(tlc: TierLevelConfig):
        if tracking is not None:
            tracking.setdefault("factory_calls", []).append(tlc.model_name)

        def fn():
            if counter["remaining"] > 0:
                counter["remaining"] -= 1
                if isinstance(exception, type):
                    raise exception("boom")
                raise exception
            return "finally-ok"

        return fn

    return factory


def _exhausted_failing_factory(
    exception: type[Exception] | Exception = RuntimeError,
    *,
    tracking: dict | None = None,
):
    """Return a factory whose callable always raises."""

    def factory(tlc: TierLevelConfig):
        if tracking is not None:
            tracking.setdefault("factory_calls", []).append(tlc.model_name)

        def fn():
            if isinstance(exception, type):
                raise exception("boom")
            raise exception

        return fn

    return factory


# --------------------------------------------------------------------------- #
#  _next_unvisited_tier                                                       #
# --------------------------------------------------------------------------- #


def test_next_unvisited_from_level1_no_visited():
    assert _next_unvisited_tier(TierLevel.LEVEL1, frozenset()) == TierLevel.LEVEL2


def test_next_unvisited_from_level1_level2_visited():
    assert (
        _next_unvisited_tier(TierLevel.LEVEL1, frozenset({TierLevel.LEVEL2}))
        == TierLevel.LEVEL3
    )


def test_next_unvisited_from_level1_all_higher_visited_returns_lower_none():
    # No lower tiers for LEVEL1, and all higher are visited → None
    assert (
        _next_unvisited_tier(
            TierLevel.LEVEL1,
            frozenset({TierLevel.LEVEL2, TierLevel.LEVEL3}),
        )
        is None
    )


def test_next_unvisited_from_level2_prefers_higher():
    # LEVEL2 → LEVEL3 first (higher), then LEVEL1 (lower)
    assert _next_unvisited_tier(TierLevel.LEVEL2, frozenset()) == TierLevel.LEVEL3


def test_next_unvisited_from_level2_higher_visited_returns_lower():
    assert (
        _next_unvisited_tier(TierLevel.LEVEL2, frozenset({TierLevel.LEVEL3}))
        == TierLevel.LEVEL1
    )


def test_next_unvisited_from_level2_all_others_visited_returns_none():
    assert (
        _next_unvisited_tier(
            TierLevel.LEVEL2,
            frozenset({TierLevel.LEVEL1, TierLevel.LEVEL3}),
        )
        is None
    )


def test_next_unvisited_from_level3_only_lower():
    # LEVEL3 → LEVEL2 (highest lower), then LEVEL1
    assert _next_unvisited_tier(TierLevel.LEVEL3, frozenset()) == TierLevel.LEVEL2


def test_next_unvisited_from_level3_level2_visited_returns_level1():
    assert (
        _next_unvisited_tier(TierLevel.LEVEL3, frozenset({TierLevel.LEVEL2}))
        == TierLevel.LEVEL1
    )


def test_next_unvisited_from_level3_all_visited_returns_none():
    assert (
        _next_unvisited_tier(
            TierLevel.LEVEL3,
            frozenset({TierLevel.LEVEL1, TierLevel.LEVEL2}),
        )
        is None
    )


def test_next_unvisited_unknown_level_returns_none():
    # Defensive: if somehow a level not in _ALL_TIER_LEVELS is passed
    class UnknownLevel:
        value = "unknown"

    assert _next_unvisited_tier(UnknownLevel(), frozenset()) is None


# --------------------------------------------------------------------------- #
#  call_with_tier_fallback                                                    #
# --------------------------------------------------------------------------- #


def test_successful_call_returns_result():
    out = call_with_tier_fallback(
        _factory_that_succeeds("hello"),
        tier_config=_STD_TIER_CONFIG,
        sleep=_noop_sleep,
    )
    assert out == "hello"


def test_fallback_disabled_raises_immediately():
    with pytest.raises(RuntimeError, match="boom"):
        call_with_tier_fallback(
            _exhausted_failing_factory(RuntimeError),
            tier_config=_STD_TIER_CONFIG,
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
        tier_config=_STD_TIER_CONFIG,
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
        tier_config=_STD_TIER_CONFIG,
        fallback_enabled=True,
        max_fallback_depth=2,
        sleep=_noop_sleep,
    )
    assert out == "l3-ok"
    assert tracking["factory_calls"] == ["opus", "haiku", "sonnet"]


def test_fallback_level2_to_level3_to_level1():
    """Start at level2: level2 fails → level3 fails → level1 succeeds.
    Demonstrates bidirectional fallback: higher first, then lower."""
    tracking: dict = {}
    counter = {"remaining": 2}

    def factory(tlc: TierLevelConfig):
        tracking.setdefault("factory_calls", []).append(tlc.model_name)

        def fn():
            if counter["remaining"] > 0:
                counter["remaining"] -= 1
                raise RuntimeError(f"fail-{tlc.model_name}")
            return "l1-ok"

        return fn

    out = call_with_tier_fallback(
        factory,
        tier_config=_STD_TIER_CONFIG,
        level=TierLevel.LEVEL2,
        fallback_enabled=True,
        max_fallback_depth=2,
        sleep=_noop_sleep,
    )
    assert out == "l1-ok"
    assert tracking["factory_calls"] == ["haiku", "sonnet", "opus"]


def test_fallback_level3_to_level2_to_level1():
    """Start at level3: level3 fails → level2 fails → level1 succeeds.
    Only lower tiers available."""
    tracking: dict = {}
    counter = {"remaining": 2}

    def factory(tlc: TierLevelConfig):
        tracking.setdefault("factory_calls", []).append(tlc.model_name)

        def fn():
            if counter["remaining"] > 0:
                counter["remaining"] -= 1
                raise RuntimeError(f"fail-{tlc.model_name}")
            return "l1-ok"

        return fn

    out = call_with_tier_fallback(
        factory,
        tier_config=_STD_TIER_CONFIG,
        level=TierLevel.LEVEL3,
        fallback_enabled=True,
        max_fallback_depth=2,
        sleep=_noop_sleep,
    )
    assert out == "l1-ok"
    assert tracking["factory_calls"] == ["sonnet", "haiku", "opus"]


def test_exhausted_all_levels_reraises_last_error():
    """All three levels fail → last exception re-raised."""
    with pytest.raises(RuntimeError, match="boom"):
        call_with_tier_fallback(
            _exhausted_failing_factory(RuntimeError),
            tier_config=_STD_TIER_CONFIG,
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
        tier_config=_STD_TIER_CONFIG,
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
            tier_config=_STD_TIER_CONFIG,
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
            tier_config=_STD_TIER_CONFIG,
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
        tier_config=_STD_TIER_CONFIG,
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
            tier_config=_STD_TIER_CONFIG,
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
        tier_config=_STD_TIER_CONFIG,
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
        _factory_that_succeeds("ok"),
        tier_config=_STD_TIER_CONFIG,
        what="happy-path",
        sleep=_noop_sleep,
    )

    warn_messages = [r.message for r in caplog.records if r.levelno == logging.WARNING]
    assert len(warn_messages) == 0


# --------------------------------------------------------------------------- #
#  acall_with_tier_fallback (async mirror)                                    #
# --------------------------------------------------------------------------- #


def test_acall_successful_call_returns_result():
    def factory(tlc: TierLevelConfig):
        async def fn():
            return "async-ok"

        return fn

    out = asyncio.run(
        acall_with_tier_fallback(
            factory,
            tier_config=_STD_TIER_CONFIG,
            sleep=_anoop_sleep,
        )
    )
    assert out == "async-ok"


def test_acall_fallback_disabled_raises_immediately():
    def factory(tlc: TierLevelConfig):
        async def fn():
            raise RuntimeError("async-boom")

        return fn

    with pytest.raises(RuntimeError, match="async-boom"):
        asyncio.run(
            acall_with_tier_fallback(
                factory,
                tier_config=_STD_TIER_CONFIG,
                fallback_enabled=False,
                sleep=_anoop_sleep,
            )
        )


def test_acall_fallback_level1_to_level2_to_level3():
    """Async: level1 fails, level2 fails, level3 succeeds."""
    tracking: dict = {}
    counter = {"remaining": 2}

    def factory(tlc: TierLevelConfig):
        tracking.setdefault("factory_calls", []).append(tlc.model_name)

        async def fn():
            if counter["remaining"] > 0:
                counter["remaining"] -= 1
                raise RuntimeError(f"fail-{tlc.model_name}")
            return "l3-ok"

        return fn

    out = asyncio.run(
        acall_with_tier_fallback(
            factory,
            tier_config=_STD_TIER_CONFIG,
            fallback_enabled=True,
            max_fallback_depth=2,
            sleep=_anoop_sleep,
        )
    )
    assert out == "l3-ok"
    assert tracking["factory_calls"] == ["opus", "haiku", "sonnet"]


def test_acall_fallback_level2_to_level3_to_level1():
    """Async bidirectional: start at level2 → level3 → level1."""
    tracking: dict = {}
    counter = {"remaining": 2}

    def factory(tlc: TierLevelConfig):
        tracking.setdefault("factory_calls", []).append(tlc.model_name)

        async def fn():
            if counter["remaining"] > 0:
                counter["remaining"] -= 1
                raise RuntimeError(f"fail-{tlc.model_name}")
            return "l1-ok"

        return fn

    out = asyncio.run(
        acall_with_tier_fallback(
            factory,
            tier_config=_STD_TIER_CONFIG,
            level=TierLevel.LEVEL2,
            fallback_enabled=True,
            max_fallback_depth=2,
            sleep=_anoop_sleep,
        )
    )
    assert out == "l1-ok"
    assert tracking["factory_calls"] == ["haiku", "sonnet", "opus"]


def test_acall_exhausted_all_levels_reraises_last_error():
    def factory(tlc: TierLevelConfig):
        async def fn():
            raise RuntimeError("async-exhausted")

        return fn

    with pytest.raises(RuntimeError, match="async-exhausted"):
        asyncio.run(
            acall_with_tier_fallback(
                factory,
                tier_config=_STD_TIER_CONFIG,
                fallback_enabled=True,
                max_fallback_depth=2,
                sleep=_anoop_sleep,
            )
        )


def test_acall_max_fallback_depth_limits_promotions():
    """Async: max_fallback_depth=1 limits promotions."""
    tracking: dict = {}

    def factory(tlc: TierLevelConfig):
        tracking.setdefault("factory_calls", []).append(tlc.model_name)

        async def fn():
            raise RuntimeError(f"fail-{tlc.model_name}")

        return fn

    with pytest.raises(RuntimeError, match="fail-haiku"):
        asyncio.run(
            acall_with_tier_fallback(
                factory,
                tier_config=_STD_TIER_CONFIG,
                fallback_enabled=True,
                max_fallback_depth=1,
                sleep=_anoop_sleep,
            )
        )
    assert tracking["factory_calls"] == ["opus", "haiku"]


def test_acall_logging_output(caplog):
    """Async: INFO/WARNING logs emitted correctly."""
    caplog.set_level(logging.INFO, logger="robotsix_llmio.tier_fallback")

    counter = {"remaining": 1}

    def factory(tlc: TierLevelConfig):
        async def fn():
            if counter["remaining"] > 0:
                counter["remaining"] -= 1
                raise RuntimeError("async-test-fail")
            return "ok"

        return fn

    asyncio.run(
        acall_with_tier_fallback(
            factory,
            tier_config=_STD_TIER_CONFIG,
            fallback_enabled=True,
            max_fallback_depth=2,
            what="async-op",
            sleep=_anoop_sleep,
        )
    )

    info_messages = [r.message for r in caplog.records if r.levelno == logging.INFO]
    assert any(
        "async-op: trying level1 (provider=claudeSDK, model=opus)" in msg
        for msg in info_messages
    )
    assert any(
        "async-op: trying level2 (provider=claudeSDK, model=haiku)" in msg
        for msg in info_messages
    )
    assert any("async-op: level2 succeeded" in msg for msg in info_messages)

    warn_messages = [r.message for r in caplog.records if r.levelno == logging.WARNING]
    assert any(
        "async-op: level1 failed with RuntimeError — falling back to level2" in msg
        for msg in warn_messages
    )

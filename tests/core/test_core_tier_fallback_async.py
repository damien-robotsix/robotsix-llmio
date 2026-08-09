"""Async ``acall_with_tier_fallback`` tests.

Extracted from ``test_core_tier_fallback.py`` (#20260727T132934Z).
"""

from __future__ import annotations

import asyncio
import logging

import pytest
from conftest import (
    STD_TIER_CONFIG,
    _anoop_sleep,
)

from robotsix_llmio.config.tier import TierLevel, TierLevelConfig
from robotsix_llmio.core.cooldown import reset_health_tracker
from robotsix_llmio.core.tier_fallback import acall_with_tier_fallback

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
            tier_config=STD_TIER_CONFIG,
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
                tier_config=STD_TIER_CONFIG,
                fallback_enabled=False,
                sleep=_anoop_sleep,
            )
        )


def test_acall_fallback_is_on_by_default():
    """Async mirror of the default-on guard — no ``fallback_enabled`` passed."""
    reset_health_tracker()
    tracking: dict = {}

    def factory(tlc: TierLevelConfig):
        tracking.setdefault("factory_calls", []).append(tlc.model_name)

        async def fn():
            if tlc.model_name == "opus":  # STD_TIER_CONFIG's level1
                raise RuntimeError("starting-tier-unavailable")
            return "next-tier-ok"

        return fn

    out = asyncio.run(
        acall_with_tier_fallback(
            factory,
            tier_config=STD_TIER_CONFIG,
            sleep=_anoop_sleep,
        )
    )
    assert out == "next-tier-ok"
    assert tracking["factory_calls"] == ["opus", "haiku"]


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
            tier_config=STD_TIER_CONFIG,
            fallback_enabled=True,
            max_fallback_depth=2,
            sleep=_anoop_sleep,
        )
    )
    assert out == "l3-ok"
    assert tracking["factory_calls"] == ["opus", "haiku", "sonnet"]


def test_acall_fallback_level2_to_level3_to_level4():
    """Async escalation: start at level2 → level3 → level4."""
    tracking: dict = {}
    counter = {"remaining": 2}

    def factory(tlc: TierLevelConfig):
        tracking.setdefault("factory_calls", []).append(tlc.model_name)

        async def fn():
            if counter["remaining"] > 0:
                counter["remaining"] -= 1
                raise RuntimeError(f"fail-{tlc.model_name}")
            return "l4-ok"

        return fn

    out = asyncio.run(
        acall_with_tier_fallback(
            factory,
            tier_config=STD_TIER_CONFIG,
            level=TierLevel.LEVEL2,
            fallback_enabled=True,
            max_fallback_depth=2,
            sleep=_anoop_sleep,
        )
    )
    assert out == "l4-ok"
    assert tracking["factory_calls"] == ["haiku", "sonnet", "claude-fable-5"]


def test_acall_exhausted_all_levels_reraises_last_error():
    def factory(tlc: TierLevelConfig):
        async def fn():
            raise RuntimeError("async-exhausted")

        return fn

    with pytest.raises(RuntimeError, match="async-exhausted"):
        asyncio.run(
            acall_with_tier_fallback(
                factory,
                tier_config=STD_TIER_CONFIG,
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
                tier_config=STD_TIER_CONFIG,
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
            tier_config=STD_TIER_CONFIG,
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

"""Provider-exhaustion skip-same-provider tests for the tier-fallback loop.

When a level fails with a *provider-wide* exhaustion
(:class:`~robotsix_llmio.exceptions.ProviderExhaustedError`, e.g.
``ClaudeSDKUsageExhaustedError``) every sibling level backed by the same
provider shares the exhausted capacity, so the loop skips them all in one step
instead of wasting fallback hops walking tiers that would only fail the same
way. Ordinary failures (transient errors, rate limits) still fall back one
level at a time.
"""

from __future__ import annotations

import asyncio

from conftest import STD_TIER_CONFIG, _anoop_sleep

from robotsix_llmio.claude_sdk import ClaudeSDKUsageExhaustedError
from robotsix_llmio.config.tier import (
    TierConfig,
    TierLevel,
    TierLevelConfig,
)
from robotsix_llmio.core.cooldown import reset_health_tracker
from robotsix_llmio.core.tier_fallback import acall_with_tier_fallback

# A mixed-provider tier config: Claude siblings interleaved with OpenRouter
# levels so a Claude exhaustion must *skip* the sibling Claude tiers and land
# on an OpenRouter tier.
#
#   level1 → openrouter   level2 → claudeSDK   level3 → openrouter
#   level4 → claudeSDK    level5 → claudeSDK
MIXED_TIER_CONFIG = TierConfig(
    level1=TierLevelConfig(model="openrouter-cheap"),
    level2=TierLevelConfig(model="claudeSDK-haiku"),
    level3=TierLevelConfig(model="openrouter-mid"),
    level4=TierLevelConfig(model="claudeSDK-sonnet"),
    level5=TierLevelConfig(model="claudeSDK-frontier"),
)


def test_provider_exhaustion_skips_all_same_provider_levels():
    """A Claude exhaustion skips every sibling Claude tier in ONE hop.

    Starting at level5 (Claude) with ``max_fallback_depth=1``: without the
    skip, a single promotion would only reach level4 (also Claude, also
    exhausted) and then raise. With the skip, level5/level4/level2 (all Claude)
    are marked visited for free and the one promotion lands on level3
    (OpenRouter), which succeeds.
    """
    reset_health_tracker()
    tracking: dict = {}

    def factory(tlc: TierLevelConfig):
        tracking.setdefault("factory_calls", []).append(tlc.model_name)

        async def fn():
            if tlc.provider == "claudeSDK":
                raise ClaudeSDKUsageExhaustedError("out of usage credits")
            return "openrouter-ok"

        return fn

    out = asyncio.run(
        acall_with_tier_fallback(
            factory,
            tier_config=MIXED_TIER_CONFIG,
            level=TierLevel.LEVEL5,
            fallback_enabled=True,
            max_fallback_depth=1,
            sleep=_anoop_sleep,
        )
    )
    assert out == "openrouter-ok"
    # frontier (level5) fails → skip haiku/sonnet → land on mid (level3).
    assert tracking["factory_calls"] == ["frontier", "mid"]
    assert "haiku" not in tracking["factory_calls"]
    assert "sonnet" not in tracking["factory_calls"]


def test_mixed_provider_walk_skips_only_matching_provider():
    """Claude siblings are skipped, but OpenRouter tiers are still walked.

    level5 (Claude) exhausts → skip Claude siblings → level3 (OpenRouter) is
    tried and fails with a *non-exhaustion* transient error → the loop walks
    normally to the next OpenRouter tier (level1), which succeeds. The skipped
    Claude tiers (haiku/sonnet) are never invoked.
    """
    reset_health_tracker()
    tracking: dict = {}

    def factory(tlc: TierLevelConfig):
        tracking.setdefault("factory_calls", []).append(tlc.model_name)

        async def fn():
            if tlc.provider == "claudeSDK":
                raise ClaudeSDKUsageExhaustedError("out of usage credits")
            if tlc.model_name == "mid":
                raise RuntimeError("transient-openrouter-blip")
            return "openrouter-cheap-ok"

        return fn

    out = asyncio.run(
        acall_with_tier_fallback(
            factory,
            tier_config=MIXED_TIER_CONFIG,
            level=TierLevel.LEVEL5,
            fallback_enabled=True,
            max_fallback_depth=2,
            sleep=_anoop_sleep,
        )
    )
    assert out == "openrouter-cheap-ok"
    assert tracking["factory_calls"] == ["frontier", "mid", "cheap"]
    assert "haiku" not in tracking["factory_calls"]
    assert "sonnet" not in tracking["factory_calls"]


def test_non_exhaustion_failure_falls_back_one_level_at_a_time():
    """A plain (non-exhaustion) error does NOT trigger the same-provider skip.

    With an all-Claude config, a plain ``RuntimeError`` walks level-by-level
    through the sibling Claude tiers exactly as before — proving the skip is
    gated on provider exhaustion, not on any failure.
    """
    reset_health_tracker()
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
    # All three sibling Claude tiers are walked one at a time — no skipping.
    assert tracking["factory_calls"] == ["opus", "haiku", "sonnet"]

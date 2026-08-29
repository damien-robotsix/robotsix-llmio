"""Tests for ``TierConfig`` — construction, defaults, model_validate, and
``for_level()`` resolution."""

from __future__ import annotations

import pytest

from robotsix_llmio.config.tier import (
    LEVEL1_DEFAULT,
    LEVEL2_DEFAULT,
    LEVEL3_DEFAULT,
    LEVEL4_DEFAULT,
    TierConfig,
    TierLevelConfig,
)

# ========================================================================== #
#  TierConfig
# ========================================================================== #


def test_tier_config_full_construction():
    """Explicitly providing all three levels uses those values."""
    cfg = TierConfig(
        level1=TierLevelConfig(model="claudeSDK-haiku"),
        level2=TierLevelConfig(model="openrouter-deepseek/deepseek-v4-pro"),
        level3=TierLevelConfig(model="claudeSDK-opus"),
    )
    assert cfg.level1.model == "claudeSDK-haiku"
    assert cfg.level2.model == "openrouter-deepseek/deepseek-v4-pro"
    assert cfg.level3.model == "claudeSDK-opus"


def test_tier_config_defaults_when_omitted():
    """Constructing with only ``level1`` falls back to baked defaults for
    ``level2``, ``level3``, and ``level4``."""
    cfg = TierConfig(
        level1=TierLevelConfig(model="claudeSDK-haiku"),
    )
    assert cfg.level1.model == "claudeSDK-haiku"
    assert cfg.level2 == LEVEL2_DEFAULT
    assert cfg.level3 == LEVEL3_DEFAULT
    assert cfg.level4 == LEVEL4_DEFAULT


def test_tier_config_all_defaults():
    """``TierConfig(level1=LEVEL1_DEFAULT)`` gives baked defaults for L2/L3/L4."""
    cfg = TierConfig(level1=LEVEL1_DEFAULT)
    assert cfg.level2 == LEVEL2_DEFAULT
    assert cfg.level3 == LEVEL3_DEFAULT
    assert cfg.level4 == LEVEL4_DEFAULT


def test_tier_config_partial_override_level2():
    """Specifying ``level2`` overrides the default; ``level3`` stays baked."""
    custom = TierLevelConfig(model="claudeSDK-sonnet")
    cfg = TierConfig(level1=LEVEL1_DEFAULT, level2=custom)
    assert cfg.level2 is custom
    assert cfg.level3 == LEVEL3_DEFAULT


def test_tier_config_partial_override_level3():
    """Specifying ``level3`` overrides the default; ``level2`` stays baked."""
    custom = TierLevelConfig(model="claudeSDK-sonnet")
    cfg = TierConfig(level1=LEVEL1_DEFAULT, level3=custom)
    assert cfg.level3 is custom
    assert cfg.level2 == LEVEL2_DEFAULT


def test_tier_config_partial_override_level4():
    """Specifying ``level4`` overrides the default; ``level3`` stays baked."""
    custom = TierLevelConfig(model="claudeSDK-opus")
    cfg = TierConfig(level1=LEVEL1_DEFAULT, level4=custom)
    assert cfg.level4 is custom
    assert cfg.level3 == LEVEL3_DEFAULT


def test_tier_config_model_validate_from_dict():
    """``model_validate`` from a plain dict populates all tiers, applying
    baked defaults for omitted ones."""
    data = {
        "level1": {"model": "claudeSDK-haiku"},
    }
    cfg = TierConfig.model_validate(data)
    assert cfg.level1.model == "claudeSDK-haiku"
    assert cfg.level1.provider == "claudeSDK"
    assert cfg.level2 == LEVEL2_DEFAULT
    assert cfg.level3 == LEVEL3_DEFAULT
    assert cfg.level4 == LEVEL4_DEFAULT


def test_tier_config_model_validate_full_dict():
    """All four tiers can be supplied in the dict."""
    data = {
        "level1": {"model": "claudeSDK-haiku"},
        "level2": {"model": "openrouter-deepseek/deepseek-v4-pro"},
        "level3": {"model": "claudeSDK-opus"},
        "level4": {"model": "claudeSDK-claude-fable-5"},
    }
    cfg = TierConfig.model_validate(data)
    assert cfg.level1.model == "claudeSDK-haiku"
    assert cfg.level2.model == "openrouter-deepseek/deepseek-v4-pro"
    assert cfg.level3.model == "claudeSDK-opus"
    assert cfg.level4.model == "claudeSDK-claude-fable-5"


def test_tier_config_omitting_level1_uses_default():
    """``level1`` falls back to ``LEVEL1_DEFAULT`` when omitted, so an empty
    config validates to the fully baked default."""
    cfg = TierConfig.model_validate({})
    assert cfg.level1 == LEVEL1_DEFAULT
    assert cfg.level2 == LEVEL2_DEFAULT
    assert cfg.level3 == LEVEL3_DEFAULT
    assert cfg.level4 == LEVEL4_DEFAULT


def test_tier_config_model_dump_round_trip():
    """Full config survives ``model_dump()`` → ``model_validate()``."""
    original = TierConfig(
        level1=TierLevelConfig(model="claudeSDK-opus", provider_kwargs={"x": 1}),
        level2=TierLevelConfig(model="openrouter-deepseek/deepseek-v4-pro"),
        level3=TierLevelConfig(model="claudeSDK-haiku"),
    )
    reloaded = TierConfig.model_validate(original.model_dump())
    assert reloaded == original


def test_tier_config_provider_kwargs_serialisation():
    """``provider_kwargs`` survives a full dump→validate round-trip."""
    cfg = TierConfig(
        level1=TierLevelConfig(
            model="claudeSDK-opus",
            provider_kwargs={"base_url": "https://example.com", "timeout": 30},
        ),
    )
    reloaded = TierConfig.model_validate(cfg.model_dump())
    expected = {"base_url": "https://example.com", "timeout": 30}
    assert reloaded.level1.provider_kwargs == expected


# ========================================================================== #
#  TierConfig.for_level()
# ========================================================================== #


def test_for_level_1_returns_level1():
    """``for_level(1)`` returns ``self.level1``."""
    cfg = TierConfig(
        level1=TierLevelConfig(model="claudeSDK-haiku"),
    )
    result = cfg.for_level(1)
    assert result is cfg.level1
    assert result.model == "claudeSDK-haiku"
    assert result.provider == "claudeSDK"


def test_for_level_2_returns_level2():
    """``for_level(2)`` returns ``self.level2`` — explicit or default."""
    cfg = TierConfig(
        level1=TierLevelConfig(model="claudeSDK-haiku"),
        level2=TierLevelConfig(model="openrouter-deepseek/deepseek-v4-pro"),
    )
    result = cfg.for_level(2)
    assert result is cfg.level2
    assert result.model == "openrouter-deepseek/deepseek-v4-pro"
    assert result.provider == "openrouter"


def test_for_level_3_returns_level3():
    """``for_level(3)`` returns ``self.level3`` — explicit or default."""
    cfg = TierConfig(
        level1=TierLevelConfig(model="claudeSDK-haiku"),
        level3=TierLevelConfig(model="claudeSDK-opus"),
    )
    result = cfg.for_level(3)
    assert result is cfg.level3
    assert result.model == "claudeSDK-opus"
    assert result.provider == "claudeSDK"


def test_for_level_4_returns_level4():
    """``for_level(4)`` returns ``self.level4`` — explicit or default."""
    cfg = TierConfig(
        level1=TierLevelConfig(model="claudeSDK-haiku"),
        level4=TierLevelConfig(model="claudeSDK-claude-fable-5"),
    )
    result = cfg.for_level(4)
    assert result is cfg.level4
    assert result.model == "claudeSDK-claude-fable-5"
    assert result.provider == "claudeSDK"


def test_for_level_0_raises():
    """``for_level(0)`` raises ValueError."""
    cfg = TierConfig(level1=TierLevelConfig(model="claudeSDK-haiku"))
    with pytest.raises(ValueError, match=r"`level` must be 1, 2, 3, 4, or 5, got 0"):
        cfg.for_level(0)


def test_for_level_6_raises():
    """``for_level(6)`` raises ValueError."""
    cfg = TierConfig(level1=TierLevelConfig(model="claudeSDK-haiku"))
    with pytest.raises(ValueError, match=r"`level` must be 1, 2, 3, 4, or 5, got 6"):
        cfg.for_level(6)


def test_for_level_returns_default_level2_when_not_explicitly_set():
    """``for_level(2)`` falls back to the baked LEVEL2_DEFAULT when level2
    is not explicitly configured."""
    cfg = TierConfig(level1=TierLevelConfig(model="claudeSDK-haiku"))
    result = cfg.for_level(2)
    assert result == LEVEL2_DEFAULT


def test_for_level_returns_default_level3_when_not_explicitly_set():
    """``for_level(3)`` falls back to the baked LEVEL3_DEFAULT when level3
    is not explicitly configured."""
    cfg = TierConfig(level1=TierLevelConfig(model="claudeSDK-haiku"))
    result = cfg.for_level(3)
    assert result == LEVEL3_DEFAULT


def test_for_level_returns_default_level4_when_not_explicitly_set():
    """``for_level(4)`` falls back to the baked LEVEL4_DEFAULT when level4
    is not explicitly configured."""
    cfg = TierConfig(level1=TierLevelConfig(model="claudeSDK-haiku"))
    result = cfg.for_level(4)
    assert result == LEVEL4_DEFAULT

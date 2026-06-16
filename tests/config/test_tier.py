"""Tests for the three-tier configuration schema.

Covers:
- ``TierLevel`` enum values and ``str`` behaviour
- ``TierLevelConfig`` construction, field types, ``model_dump()`` round-trip
- ``TierConfig`` defaults and partial overrides
- ``TierConfig.model_validate()`` from plain dicts
- ``LEGACY_TIER_MAP`` correctness
- ``provider_kwargs`` default and serialisation
"""

from __future__ import annotations

import pytest

from robotsix_llmio.config.tier import (
    LEGACY_TIER_MAP,
    LEVEL1_DEFAULT,
    LEVEL2_DEFAULT,
    LEVEL3_DEFAULT,
    TierConfig,
    TierLevel,
    TierLevelConfig,
)
from robotsix_llmio.core.provider import Tier as LegacyTier

# ========================================================================== #
#  TierLevel enum
# ========================================================================== #


def test_tier_level_values():
    """Each member carries the expected string value."""
    assert TierLevel.LEVEL1.value == "level1"
    assert TierLevel.LEVEL2.value == "level2"
    assert TierLevel.LEVEL3.value == "level3"


def test_tier_level_is_str_enum():
    """Members are instances of both ``str`` and ``StrEnum``."""
    for member in TierLevel:
        assert isinstance(member, str)
        assert isinstance(member, TierLevel)


def test_tier_level_str_comparison():
    """Members compare equal to their string values."""
    assert TierLevel.LEVEL1.value == "level1"
    assert TierLevel.LEVEL2.value == "level2"
    assert TierLevel.LEVEL3.value == "level3"


def test_tier_level_distinct_members():
    """Different members have different names and values."""
    members = list(TierLevel)
    assert len(members) == 3
    # All values are distinct.
    values = {m.value for m in members}
    assert len(values) == 3


def test_tier_level_members():
    """Only the three members exist — no extras."""
    assert {m.name for m in TierLevel} == {"LEVEL1", "LEVEL2", "LEVEL3"}
    assert len(list(TierLevel)) == 3


# ========================================================================== #
#  TierLevelConfig
# ========================================================================== #


def test_tier_level_config_minimal_construction():
    """Minimal construction requires only ``provider`` and ``model``."""
    cfg = TierLevelConfig(provider="p", model="m")
    assert cfg.provider == "p"
    assert cfg.model == "m"
    assert cfg.provider_kwargs == {}


def test_tier_level_config_with_provider_kwargs():
    """``provider_kwargs`` can be supplied explicitly."""
    cfg = TierLevelConfig(provider="p", model="m", provider_kwargs={"base_url": "https://x"})
    assert cfg.provider_kwargs == {"base_url": "https://x"}


def test_tier_level_config_provider_kwargs_defaults_to_empty_dict():
    """Omitting ``provider_kwargs`` yields ``{}``, never ``None``."""
    cfg = TierLevelConfig(provider="p", model="m")
    assert cfg.provider_kwargs == {}
    assert isinstance(cfg.provider_kwargs, dict)


def test_tier_level_config_field_types():
    """Fields are correctly typed."""
    cfg = TierLevelConfig(provider="x", model="y")
    assert isinstance(cfg.provider, str)
    assert isinstance(cfg.model, str)
    assert isinstance(cfg.provider_kwargs, dict)


def test_tier_level_config_model_dump_round_trip():
    """A constructed instance round-trips through ``model_dump()`` →
    ``TierLevelConfig(**dump)`` without losing data."""
    original = TierLevelConfig(
        provider="openrouter-deepseek",
        model="deepseek/deepseek-v4-flash",
        provider_kwargs={"base_url": "https://custom"},
    )
    reloaded = TierLevelConfig(**original.model_dump())
    assert reloaded == original
    assert reloaded.provider == original.provider
    assert reloaded.model == original.model
    assert reloaded.provider_kwargs == original.provider_kwargs


def test_tier_level_config_json_round_trip():
    """``model_dump_json()`` → ``model_validate_json()`` preserves equality."""
    original = TierLevelConfig(
        provider="openrouter-deepseek",
        model="deepseek/deepseek-v4-flash",
        provider_kwargs={"key": "val"},
    )
    json_str = original.model_dump_json()
    reloaded = TierLevelConfig.model_validate_json(json_str)
    assert reloaded == original


def test_tier_level_config_missing_provider_raises():
    """``provider`` is required — omitting it raises ValidationError."""
    with pytest.raises(ValueError):  # pydantic v2 raises ValidationError ⊆ ValueError
        TierLevelConfig(model="m")  # type: ignore[call-arg]


def test_tier_level_config_missing_model_raises():
    """``model`` is required — omitting it raises ValidationError."""
    with pytest.raises(ValueError):
        TierLevelConfig(provider="p")  # type: ignore[call-arg]


# ========================================================================== #
#  Baked defaults
# ========================================================================== #


def test_level1_default():
    assert LEVEL1_DEFAULT.provider == "openrouter-deepseek"
    assert LEVEL1_DEFAULT.model == "deepseek/deepseek-v4-flash"


def test_level2_default():
    assert LEVEL2_DEFAULT.provider == "openrouter-deepseek"
    assert LEVEL2_DEFAULT.model == "deepseek/deepseek-v4-pro"


def test_level3_default():
    assert LEVEL3_DEFAULT.provider == "claude-sdk"
    assert LEVEL3_DEFAULT.model == "opus"


# ========================================================================== #
#  TierConfig
# ========================================================================== #


def test_tier_config_full_construction():
    """Explicitly providing all three levels uses those values."""
    cfg = TierConfig(
        level1=TierLevelConfig(provider="a", model="1"),
        level2=TierLevelConfig(provider="b", model="2"),
        level3=TierLevelConfig(provider="c", model="3"),
    )
    assert cfg.level1.provider == "a"
    assert cfg.level2.provider == "b"
    assert cfg.level3.provider == "c"


def test_tier_config_defaults_when_omitted():
    """Constructing with only ``level1`` falls back to baked defaults for
    ``level2`` and ``level3``."""
    cfg = TierConfig(
        level1=TierLevelConfig(provider="custom-p", model="custom-m"),
    )
    assert cfg.level1.provider == "custom-p"
    assert cfg.level2 == LEVEL2_DEFAULT
    assert cfg.level3 == LEVEL3_DEFAULT


def test_tier_config_all_defaults():
    """``TierConfig(level1=LEVEL1_DEFAULT)`` gives baked defaults for L2/L3."""
    cfg = TierConfig(level1=LEVEL1_DEFAULT)
    assert cfg.level2 == LEVEL2_DEFAULT
    assert cfg.level3 == LEVEL3_DEFAULT


def test_tier_config_partial_override_level2():
    """Specifying ``level2`` overrides the default; ``level3`` stays baked."""
    custom = TierLevelConfig(provider="x", model="y")
    cfg = TierConfig(level1=LEVEL1_DEFAULT, level2=custom)
    assert cfg.level2 is custom
    assert cfg.level3 == LEVEL3_DEFAULT


def test_tier_config_partial_override_level3():
    """Specifying ``level3`` overrides the default; ``level2`` stays baked."""
    custom = TierLevelConfig(provider="x", model="y")
    cfg = TierConfig(level1=LEVEL1_DEFAULT, level3=custom)
    assert cfg.level3 is custom
    assert cfg.level2 == LEVEL2_DEFAULT


def test_tier_config_model_validate_from_dict():
    """``model_validate`` from a plain dict populates all tiers, applying
    baked defaults for omitted ones."""
    data = {
        "level1": {"provider": "p1", "model": "m1"},
    }
    cfg = TierConfig.model_validate(data)
    assert cfg.level1.provider == "p1"
    assert cfg.level1.model == "m1"
    assert cfg.level2 == LEVEL2_DEFAULT
    assert cfg.level3 == LEVEL3_DEFAULT


def test_tier_config_model_validate_full_dict():
    """All three tiers can be supplied in the dict."""
    data = {
        "level1": {"provider": "a", "model": "1"},
        "level2": {"provider": "b", "model": "2"},
        "level3": {"provider": "c", "model": "3"},
    }
    cfg = TierConfig.model_validate(data)
    assert cfg.level1.provider == "a"
    assert cfg.level2.provider == "b"
    assert cfg.level3.provider == "c"


def test_tier_config_missing_level1_raises():
    """``level1`` is required — omitting it from the dict raises."""
    with pytest.raises(ValueError):
        TierConfig.model_validate({})


def test_tier_config_model_dump_round_trip():
    """Full config survives ``model_dump()`` → ``model_validate()``."""
    original = TierConfig(
        level1=TierLevelConfig(provider="p", model="m", provider_kwargs={"x": 1}),
        level2=TierLevelConfig(provider="q", model="n"),
        level3=TierLevelConfig(provider="r", model="o"),
    )
    reloaded = TierConfig.model_validate(original.model_dump())
    assert reloaded == original


def test_tier_config_provider_kwargs_serialisation():
    """``provider_kwargs`` survives a full dump→validate round-trip."""
    cfg = TierConfig(
        level1=TierLevelConfig(
            provider="p",
            model="m",
            provider_kwargs={"base_url": "https://example.com", "timeout": 30},
        ),
    )
    reloaded = TierConfig.model_validate(cfg.model_dump())
    expected = {"base_url": "https://example.com", "timeout": 30}
    assert reloaded.level1.provider_kwargs == expected


# ========================================================================== #
#  LEGACY_TIER_MAP
# ========================================================================== #


def test_legacy_tier_map_keys():
    """All legacy ``Tier`` values are keys in the mapping."""
    assert set(LEGACY_TIER_MAP.keys()) == {LegacyTier.CHEAP, LegacyTier.DEFAULT}


def test_legacy_tier_map_cheap_to_level1():
    """``Tier.CHEAP`` maps to ``TierLevel.LEVEL1``."""
    assert LEGACY_TIER_MAP[LegacyTier.CHEAP] == TierLevel.LEVEL1


def test_legacy_tier_map_default_to_level2():
    """``Tier.DEFAULT`` maps to ``TierLevel.LEVEL2``."""
    assert LEGACY_TIER_MAP[LegacyTier.DEFAULT] == TierLevel.LEVEL2


def test_legacy_tier_map_values_are_tier_level():
    """Every value in the mapping is a ``TierLevel`` member."""
    for v in LEGACY_TIER_MAP.values():
        assert isinstance(v, TierLevel)


# ========================================================================== #
#  Re-exports from robotsix_llmio.core
# ========================================================================== #


def test_core_reexports_tier_level():
    """``TierLevel`` is importable from ``robotsix_llmio.core``."""
    from robotsix_llmio.core import TierLevel as TL

    assert TL is TierLevel


def test_core_reexports_tier_config():
    """``TierConfig`` is importable from ``robotsix_llmio.core``."""
    from robotsix_llmio.core import TierConfig as TC

    assert TC is TierConfig


def test_core_reexports_tier_level_config():
    """``TierLevelConfig`` is importable from ``robotsix_llmio.core``."""
    from robotsix_llmio.core import TierLevelConfig as TLC

    assert TLC is TierLevelConfig


def test_core_reexports_defaults():
    """The three baked defaults are importable from ``robotsix_llmio.core``."""
    from robotsix_llmio.core import (
        LEVEL1_DEFAULT as L1D,
    )
    from robotsix_llmio.core import (
        LEVEL2_DEFAULT as L2D,
    )
    from robotsix_llmio.core import (
        LEVEL3_DEFAULT as L3D,
    )

    assert L1D is LEVEL1_DEFAULT
    assert L2D is LEVEL2_DEFAULT
    assert L3D is LEVEL3_DEFAULT


def test_core_reexports_legacy_tier_map():
    """``LEGACY_TIER_MAP`` is importable from ``robotsix_llmio.core``."""
    from robotsix_llmio.core import LEGACY_TIER_MAP as LTM

    assert LTM is LEGACY_TIER_MAP

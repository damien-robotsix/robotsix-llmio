"""Tests for the three-tier configuration schema.

Covers:
- ``TierLevel`` enum values and ``str`` behaviour
- ``TierLevelConfig`` construction, field types, ``model_dump()`` round-trip
- ``transport`` validation and the backward-compatible ``provider`` shape
- ``TierConfig`` defaults and partial overrides
- ``TierConfig.model_validate()`` from plain dicts
- ``TierConfig.for_level()`` integer→TierLevelConfig resolution
- ``LEGACY_TIER_MAP`` correctness
- ``provider_kwargs`` default and serialisation
"""

from __future__ import annotations

import pytest

from robotsix_llmio.config.model_registry import UnknownModelError
from robotsix_llmio.config.tier import (
    LEGACY_TIER_MAP,
    LEVEL1_DEFAULT,
    LEVEL2_DEFAULT,
    LEVEL3_DEFAULT,
    TierConfig,
    TierLevel,
    TierLevelConfig,
)
from robotsix_llmio.config.transport import UnknownTransportError
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
    """Minimal construction requires only ``transport`` and ``model``."""
    cfg = TierLevelConfig(transport="claude-sdk", model="opus")
    assert cfg.transport == "claude-sdk"
    assert cfg.model == "opus"
    assert cfg.provider_kwargs == {}


def test_tier_level_config_valid_transport_and_model():
    """A valid transport alias with a supported model constructs cleanly."""
    cfg = TierLevelConfig(
        transport="openrouter[deepseek]", model="deepseek/deepseek-v4-pro"
    )
    assert cfg.transport == "openrouter[deepseek]"
    assert cfg.provider == "openrouter-deepseek"


def test_tier_level_config_with_provider_kwargs():
    """``provider_kwargs`` can be supplied explicitly."""
    cfg = TierLevelConfig(
        transport="claude-sdk", model="opus", provider_kwargs={"base_url": "https://x"}
    )
    assert cfg.provider_kwargs == {"base_url": "https://x"}


def test_tier_level_config_provider_kwargs_defaults_to_empty_dict():
    """Omitting ``provider_kwargs`` yields ``{}``, never ``None``."""
    cfg = TierLevelConfig(transport="claude-sdk", model="opus")
    assert cfg.provider_kwargs == {}
    assert isinstance(cfg.provider_kwargs, dict)


def test_tier_level_config_field_types():
    """Fields are correctly typed."""
    cfg = TierLevelConfig(transport="claude-sdk", model="opus")
    assert isinstance(cfg.transport, str)
    assert isinstance(cfg.model, str)
    assert isinstance(cfg.provider_kwargs, dict)


def test_tier_level_config_model_dump_round_trip():
    """A constructed instance round-trips through ``model_dump()`` →
    ``TierLevelConfig(**dump)`` without losing data."""
    original = TierLevelConfig(
        transport="openrouter[deepseek]",
        model="deepseek/deepseek-v4-flash",
        provider_kwargs={"base_url": "https://custom"},
    )
    reloaded = TierLevelConfig(**original.model_dump())
    assert reloaded == original
    assert reloaded.transport == original.transport
    assert reloaded.model == original.model
    assert reloaded.provider_kwargs == original.provider_kwargs


def test_tier_level_config_model_dump_emits_transport_not_provider():
    """``model_dump()`` emits the ``transport`` field and never ``provider``."""
    dump = TierLevelConfig(transport="claude-sdk", model="opus").model_dump()
    assert dump["transport"] == "claude-sdk"
    assert "provider" not in dump


def test_tier_level_config_json_round_trip():
    """``model_dump_json()`` → ``model_validate_json()`` preserves equality."""
    original = TierLevelConfig(
        transport="openrouter[deepseek]",
        model="deepseek/deepseek-v4-flash",
        provider_kwargs={"key": "val"},
    )
    json_str = original.model_dump_json()
    reloaded = TierLevelConfig.model_validate_json(json_str)
    assert reloaded == original


def test_tier_level_config_unknown_transport_raises():
    """An unknown transport raises :class:`UnknownTransportError`."""
    with pytest.raises(UnknownTransportError):
        TierLevelConfig(transport="not-a-transport", model="opus")


def test_tier_level_config_unsupported_model_raises():
    """A known transport with an unsupported model raises ``UnknownModelError``."""
    with pytest.raises(UnknownModelError):
        TierLevelConfig(transport="claude-sdk", model="not-a-model")


def test_tier_level_config_backward_compat_provider_kwarg():
    """Constructing with the legacy ``provider`` kwarg converts to transport."""
    cfg = TierLevelConfig(
        provider="openrouter-deepseek",  # type: ignore[call-arg]
        model="deepseek/deepseek-v4-pro",
    )
    assert cfg.transport == "openrouter[deepseek]"
    assert cfg.provider == "openrouter-deepseek"


def test_tier_level_config_backward_compat_model_validate():
    """``model_validate`` accepts the old ``provider`` shape."""
    cfg = TierLevelConfig.model_validate(
        {"provider": "openrouter-deepseek", "model": "deepseek/deepseek-v4-flash"}
    )
    assert cfg.transport == "openrouter[deepseek]"
    assert cfg.provider == "openrouter-deepseek"


def test_tier_level_config_transport_wins_over_provider():
    """When both keys are present, ``transport`` wins and ``provider`` drops."""
    cfg = TierLevelConfig.model_validate(
        {
            "transport": "claude-sdk",
            "provider": "openrouter-deepseek",
            "model": "opus",
        }
    )
    assert cfg.transport == "claude-sdk"
    assert cfg.provider == "claude-sdk"


def test_tier_level_config_missing_transport_raises():
    """``transport`` is required — omitting it raises ValidationError."""
    with pytest.raises(ValueError):  # pydantic v2 raises ValidationError ⊆ ValueError
        TierLevelConfig(model="opus")  # type: ignore[call-arg]


def test_tier_level_config_missing_model_raises():
    """``model`` is required — omitting it raises ValidationError."""
    with pytest.raises(ValueError):
        TierLevelConfig(transport="claude-sdk")  # type: ignore[call-arg]


# ========================================================================== #
#  Baked defaults
# ========================================================================== #


def test_level1_default():
    assert LEVEL1_DEFAULT.transport == "openrouter[deepseek]"
    assert LEVEL1_DEFAULT.provider == "openrouter-deepseek"
    assert LEVEL1_DEFAULT.model == "deepseek/deepseek-v4-flash"


def test_level2_default():
    assert LEVEL2_DEFAULT.transport == "openrouter[deepseek]"
    assert LEVEL2_DEFAULT.provider == "openrouter-deepseek"
    assert LEVEL2_DEFAULT.model == "deepseek/deepseek-v4-pro"


def test_level3_default():
    assert LEVEL3_DEFAULT.transport == "claude-sdk"
    assert LEVEL3_DEFAULT.provider == "claude-sdk"
    assert LEVEL3_DEFAULT.model == "opus"


# ========================================================================== #
#  TierConfig
# ========================================================================== #


def test_tier_config_full_construction():
    """Explicitly providing all three levels uses those values."""
    cfg = TierConfig(
        level1=TierLevelConfig(transport="claude-sdk", model="haiku"),
        level2=TierLevelConfig(
            transport="openrouter[deepseek]", model="deepseek/deepseek-v4-pro"
        ),
        level3=TierLevelConfig(transport="claude-sdk", model="opus"),
    )
    assert cfg.level1.model == "haiku"
    assert cfg.level2.model == "deepseek/deepseek-v4-pro"
    assert cfg.level3.model == "opus"


def test_tier_config_defaults_when_omitted():
    """Constructing with only ``level1`` falls back to baked defaults for
    ``level2`` and ``level3``."""
    cfg = TierConfig(
        level1=TierLevelConfig(transport="claude-sdk", model="haiku"),
    )
    assert cfg.level1.model == "haiku"
    assert cfg.level2 == LEVEL2_DEFAULT
    assert cfg.level3 == LEVEL3_DEFAULT


def test_tier_config_all_defaults():
    """``TierConfig(level1=LEVEL1_DEFAULT)`` gives baked defaults for L2/L3."""
    cfg = TierConfig(level1=LEVEL1_DEFAULT)
    assert cfg.level2 == LEVEL2_DEFAULT
    assert cfg.level3 == LEVEL3_DEFAULT


def test_tier_config_partial_override_level2():
    """Specifying ``level2`` overrides the default; ``level3`` stays baked."""
    custom = TierLevelConfig(transport="claude-sdk", model="sonnet")
    cfg = TierConfig(level1=LEVEL1_DEFAULT, level2=custom)
    assert cfg.level2 is custom
    assert cfg.level3 == LEVEL3_DEFAULT


def test_tier_config_partial_override_level3():
    """Specifying ``level3`` overrides the default; ``level2`` stays baked."""
    custom = TierLevelConfig(transport="claude-sdk", model="sonnet")
    cfg = TierConfig(level1=LEVEL1_DEFAULT, level3=custom)
    assert cfg.level3 is custom
    assert cfg.level2 == LEVEL2_DEFAULT


def test_tier_config_model_validate_from_dict():
    """``model_validate`` from a plain dict populates all tiers, applying
    baked defaults for omitted ones."""
    data = {
        "level1": {"transport": "claude-sdk", "model": "haiku"},
    }
    cfg = TierConfig.model_validate(data)
    assert cfg.level1.transport == "claude-sdk"
    assert cfg.level1.model == "haiku"
    assert cfg.level2 == LEVEL2_DEFAULT
    assert cfg.level3 == LEVEL3_DEFAULT


def test_tier_config_model_validate_full_dict():
    """All three tiers can be supplied in the dict."""
    data = {
        "level1": {"transport": "claude-sdk", "model": "haiku"},
        "level2": {
            "transport": "openrouter[deepseek]",
            "model": "deepseek/deepseek-v4-pro",
        },
        "level3": {"transport": "claude-sdk", "model": "opus"},
    }
    cfg = TierConfig.model_validate(data)
    assert cfg.level1.model == "haiku"
    assert cfg.level2.model == "deepseek/deepseek-v4-pro"
    assert cfg.level3.model == "opus"


def test_tier_config_model_validate_backward_compat_provider():
    """``model_validate`` accepts the legacy ``provider`` shape per tier."""
    data = {
        "level1": {
            "provider": "openrouter-deepseek",
            "model": "deepseek/deepseek-v4-flash",
        },
    }
    cfg = TierConfig.model_validate(data)
    assert cfg.level1.transport == "openrouter[deepseek]"
    assert cfg.level1.provider == "openrouter-deepseek"


def test_tier_config_missing_level1_raises():
    """``level1`` is required — omitting it from the dict raises."""
    with pytest.raises(ValueError):
        TierConfig.model_validate({})


def test_tier_config_model_dump_round_trip():
    """Full config survives ``model_dump()`` → ``model_validate()``."""
    original = TierConfig(
        level1=TierLevelConfig(
            transport="claude-sdk", model="opus", provider_kwargs={"x": 1}
        ),
        level2=TierLevelConfig(
            transport="openrouter[deepseek]", model="deepseek/deepseek-v4-pro"
        ),
        level3=TierLevelConfig(transport="claude-sdk", model="haiku"),
    )
    reloaded = TierConfig.model_validate(original.model_dump())
    assert reloaded == original


def test_tier_config_provider_kwargs_serialisation():
    """``provider_kwargs`` survives a full dump→validate round-trip."""
    cfg = TierConfig(
        level1=TierLevelConfig(
            transport="claude-sdk",
            model="opus",
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
        level1=TierLevelConfig(transport="claude-sdk", model="haiku"),
    )
    result = cfg.for_level(1)
    assert result is cfg.level1
    assert result.transport == "claude-sdk"
    assert result.model == "haiku"


def test_for_level_2_returns_level2():
    """``for_level(2)`` returns ``self.level2`` — explicit or default."""
    cfg = TierConfig(
        level1=TierLevelConfig(transport="claude-sdk", model="haiku"),
        level2=TierLevelConfig(
            transport="openrouter[deepseek]", model="deepseek/deepseek-v4-pro"
        ),
    )
    result = cfg.for_level(2)
    assert result is cfg.level2
    assert result.transport == "openrouter[deepseek]"
    assert result.model == "deepseek/deepseek-v4-pro"


def test_for_level_3_returns_level3():
    """``for_level(3)`` returns ``self.level3`` — explicit or default."""
    cfg = TierConfig(
        level1=TierLevelConfig(transport="claude-sdk", model="haiku"),
        level3=TierLevelConfig(transport="claude-sdk", model="opus"),
    )
    result = cfg.for_level(3)
    assert result is cfg.level3
    assert result.transport == "claude-sdk"
    assert result.model == "opus"


def test_for_level_0_raises():
    """``for_level(0)`` raises ValueError."""
    cfg = TierConfig(level1=TierLevelConfig(transport="claude-sdk", model="haiku"))
    with pytest.raises(ValueError, match=r"`level` must be 1, 2, or 3, got 0"):
        cfg.for_level(0)


def test_for_level_4_raises():
    """``for_level(4)`` raises ValueError."""
    cfg = TierConfig(level1=TierLevelConfig(transport="claude-sdk", model="haiku"))
    with pytest.raises(ValueError, match=r"`level` must be 1, 2, or 3, got 4"):
        cfg.for_level(4)


def test_for_level_returns_default_level2_when_not_explicitly_set():
    """``for_level(2)`` falls back to the baked LEVEL2_DEFAULT when level2
    is not explicitly configured."""
    cfg = TierConfig(level1=TierLevelConfig(transport="claude-sdk", model="haiku"))
    result = cfg.for_level(2)
    assert result == LEVEL2_DEFAULT


def test_for_level_returns_default_level3_when_not_explicitly_set():
    """``for_level(3)`` falls back to the baked LEVEL3_DEFAULT when level3
    is not explicitly configured."""
    cfg = TierConfig(level1=TierLevelConfig(transport="claude-sdk", model="haiku"))
    result = cfg.for_level(3)
    assert result == LEVEL3_DEFAULT


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
    """``LEGACY_TIER_MAP`` is importable from ``robotsix_llmio.core``
    and emits a :exc:`DeprecationWarning` on access."""
    with pytest.warns(DeprecationWarning, match="LEGACY_TIER_MAP is deprecated"):
        from robotsix_llmio.core import LEGACY_TIER_MAP as LTM
    assert LTM is LEGACY_TIER_MAP

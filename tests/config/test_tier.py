"""Tests for the three-tier configuration schema.

Covers:
- ``TierLevel`` enum values and ``str`` behaviour
- ``TierLevelConfig`` construction, field types, ``model_dump()`` round-trip
- Combined provider-model identifier validation
- ``TierConfig`` defaults and partial overrides
- ``TierConfig.model_validate()`` from plain dicts
- ``TierConfig.for_level()`` integer→TierLevelConfig resolution
- ``provider_kwargs`` default and serialisation
"""

from __future__ import annotations

import pytest

from robotsix_llmio.config.tier import (
    LEVEL1_DEFAULT,
    LEVEL2_DEFAULT,
    LEVEL3_DEFAULT,
    TierConfig,
    TierLevel,
    TierLevelConfig,
)
from robotsix_llmio.core.identifier import MalformedIdentifierError

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
    """Minimal construction requires only ``model``."""
    cfg = TierLevelConfig(model="claudeSDK-opus")
    assert cfg.model == "claudeSDK-opus"
    assert cfg.provider == "claudeSDK"
    assert cfg.model_name == "opus"
    assert cfg.provider_kwargs == {}


def test_tier_level_config_with_bracketed_qualifier():
    """A valid identifier with a bracketed qualifier constructs and parses
    cleanly — the qualifier is stripped, leaving the bare provider."""
    cfg = TierLevelConfig(model="openrouter[deepseek]-deepseek/deepseek-v4-pro")
    assert cfg.provider == "openrouter"
    assert cfg.model_name == "deepseek/deepseek-v4-pro"


def test_tier_level_config_with_provider_kwargs():
    """``provider_kwargs`` can be supplied explicitly."""
    cfg = TierLevelConfig(
        model="claudeSDK-opus", provider_kwargs={"base_url": "https://x"}
    )
    assert cfg.provider_kwargs == {"base_url": "https://x"}


def test_tier_level_config_provider_kwargs_defaults_to_empty_dict():
    """Omitting ``provider_kwargs`` yields ``{}``, never ``None``."""
    cfg = TierLevelConfig(model="claudeSDK-opus")
    assert cfg.provider_kwargs == {}
    assert isinstance(cfg.provider_kwargs, dict)


def test_tier_level_config_field_types():
    """Fields are correctly typed."""
    cfg = TierLevelConfig(model="claudeSDK-opus")
    assert isinstance(cfg.model, str)
    assert isinstance(cfg.provider_kwargs, dict)
    assert isinstance(cfg.provider, str)
    assert isinstance(cfg.model_name, str)


def test_tier_level_config_model_dump_round_trip():
    """A constructed instance round-trips through ``model_dump()`` →
    ``TierLevelConfig(**dump)`` without losing data."""
    original = TierLevelConfig(
        model="openrouter[deepseek]-deepseek/deepseek-v4-flash",
        provider_kwargs={"base_url": "https://custom"},
    )
    reloaded = TierLevelConfig(**original.model_dump())
    assert reloaded == original
    assert reloaded.model == original.model
    assert reloaded.provider_kwargs == original.provider_kwargs


def test_tier_level_config_model_dump_emits_model_not_transport():
    """``model_dump()`` emits ``model``, never ``transport`` or ``provider``."""
    dump = TierLevelConfig(model="claudeSDK-opus").model_dump()
    assert dump["model"] == "claudeSDK-opus"
    assert "transport" not in dump
    assert "provider" not in dump


def test_tier_level_config_json_round_trip():
    """``model_dump_json()`` → ``model_validate_json()`` preserves equality."""
    original = TierLevelConfig(
        model="openrouter[deepseek]-deepseek/deepseek-v4-flash",
        provider_kwargs={"key": "val"},
    )
    json_str = original.model_dump_json()
    reloaded = TierLevelConfig.model_validate_json(json_str)
    assert reloaded == original


def test_tier_level_config_unknown_provider_prefix_raises():
    """An unknown provider prefix raises :class:`ValueError`."""
    with pytest.raises(ValueError, match="Unknown provider prefix"):
        TierLevelConfig(model="bogusPrefix-opus")


def test_tier_level_config_unknown_model_succeeds():
    """A valid provider prefix with an unrecognised model name is accepted —
    model-name cross-check is the backend's concern."""
    cfg = TierLevelConfig(model="claudeSDK-not-a-model")
    assert cfg.provider == "claudeSDK"
    assert cfg.model_name == "not-a-model"


def test_tier_level_config_missing_model_raises():
    """``model`` is required — omitting it raises ValidationError."""
    with pytest.raises(ValueError):  # pydantic v2 raises ValidationError ⊆ ValueError
        TierLevelConfig()  # type: ignore[call-arg]

    with pytest.raises(ValueError):
        TierLevelConfig(provider_kwargs={})  # type: ignore[call-arg]


def test_tier_level_config_malformed_identifier_raises():
    """A malformed identifier (no hyphen) raises :class:`MalformedIdentifierError`."""
    with pytest.raises(MalformedIdentifierError):
        TierLevelConfig(model="no_hyphen_at_all")


# ========================================================================== #
#  Baked defaults
# ========================================================================== #


def test_level1_default():
    assert LEVEL1_DEFAULT.model == "openrouter[deepseek]-deepseek/deepseek-v4-flash"
    assert LEVEL1_DEFAULT.provider == "openrouter"
    assert LEVEL1_DEFAULT.model_name == "deepseek/deepseek-v4-flash"


def test_level2_default():
    assert LEVEL2_DEFAULT.model == "openrouter[deepseek]-deepseek/deepseek-v4-pro"
    assert LEVEL2_DEFAULT.provider == "openrouter"
    assert LEVEL2_DEFAULT.model_name == "deepseek/deepseek-v4-pro"


def test_level3_default():
    assert LEVEL3_DEFAULT.model == "claudeSDK-opus"
    assert LEVEL3_DEFAULT.provider == "claudeSDK"
    assert LEVEL3_DEFAULT.model_name == "opus"


# ========================================================================== #
#  TierLevelConfig parsed accessors
# ========================================================================== #


def test_tier_level_config_parsed_accessors_simple():
    """Parsed accessors work for a simple provider-model identifier."""
    cfg = TierLevelConfig(model="claudeSDK-haiku")
    assert cfg.provider == "claudeSDK"
    assert cfg.model_name == "haiku"


def test_tier_level_config_parsed_accessors_with_bracketed_qualifier():
    """Parsed accessors work for an identifier with a bracketed qualifier
    (which is stripped from the parsed provider)."""
    cfg = TierLevelConfig(model="openrouter[deepseek]-deepseek/deepseek-v4-pro")
    assert cfg.provider == "openrouter"
    assert cfg.model_name == "deepseek/deepseek-v4-pro"


def test_tier_level_config_parsed_accessors_dash_in_model_name():
    """The model_name portion may itself contain hyphens."""
    cfg = TierLevelConfig(model="claudeSDK-some-model-with-dashes")
    assert cfg.provider == "claudeSDK"
    assert cfg.model_name == "some-model-with-dashes"


# ========================================================================== #
#  TierConfig
# ========================================================================== #


def test_tier_config_full_construction():
    """Explicitly providing all three levels uses those values."""
    cfg = TierConfig(
        level1=TierLevelConfig(model="claudeSDK-haiku"),
        level2=TierLevelConfig(model="openrouter[deepseek]-deepseek/deepseek-v4-pro"),
        level3=TierLevelConfig(model="claudeSDK-opus"),
    )
    assert cfg.level1.model == "claudeSDK-haiku"
    assert cfg.level2.model == "openrouter[deepseek]-deepseek/deepseek-v4-pro"
    assert cfg.level3.model == "claudeSDK-opus"


def test_tier_config_defaults_when_omitted():
    """Constructing with only ``level1`` falls back to baked defaults for
    ``level2`` and ``level3``."""
    cfg = TierConfig(
        level1=TierLevelConfig(model="claudeSDK-haiku"),
    )
    assert cfg.level1.model == "claudeSDK-haiku"
    assert cfg.level2 == LEVEL2_DEFAULT
    assert cfg.level3 == LEVEL3_DEFAULT


def test_tier_config_all_defaults():
    """``TierConfig(level1=LEVEL1_DEFAULT)`` gives baked defaults for L2/L3."""
    cfg = TierConfig(level1=LEVEL1_DEFAULT)
    assert cfg.level2 == LEVEL2_DEFAULT
    assert cfg.level3 == LEVEL3_DEFAULT


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


def test_tier_config_model_validate_full_dict():
    """All three tiers can be supplied in the dict."""
    data = {
        "level1": {"model": "claudeSDK-haiku"},
        "level2": {"model": "openrouter[deepseek]-deepseek/deepseek-v4-pro"},
        "level3": {"model": "claudeSDK-opus"},
    }
    cfg = TierConfig.model_validate(data)
    assert cfg.level1.model == "claudeSDK-haiku"
    assert cfg.level2.model == "openrouter[deepseek]-deepseek/deepseek-v4-pro"
    assert cfg.level3.model == "claudeSDK-opus"


def test_tier_config_omitting_level1_uses_default():
    """``level1`` falls back to ``LEVEL1_DEFAULT`` when omitted, so an empty
    config validates to the fully baked default."""
    cfg = TierConfig.model_validate({})
    assert cfg.level1 == LEVEL1_DEFAULT
    assert cfg.level2 == LEVEL2_DEFAULT
    assert cfg.level3 == LEVEL3_DEFAULT


def test_tier_config_model_dump_round_trip():
    """Full config survives ``model_dump()`` → ``model_validate()``."""
    original = TierConfig(
        level1=TierLevelConfig(model="claudeSDK-opus", provider_kwargs={"x": 1}),
        level2=TierLevelConfig(model="openrouter[deepseek]-deepseek/deepseek-v4-pro"),
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
        level2=TierLevelConfig(model="openrouter[deepseek]-deepseek/deepseek-v4-pro"),
    )
    result = cfg.for_level(2)
    assert result is cfg.level2
    assert result.model == "openrouter[deepseek]-deepseek/deepseek-v4-pro"
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


def test_for_level_0_raises():
    """``for_level(0)`` raises ValueError."""
    cfg = TierConfig(level1=TierLevelConfig(model="claudeSDK-haiku"))
    with pytest.raises(ValueError, match=r"`level` must be 1, 2, or 3, got 0"):
        cfg.for_level(0)


def test_for_level_4_raises():
    """``for_level(4)`` raises ValueError."""
    cfg = TierConfig(level1=TierLevelConfig(model="claudeSDK-haiku"))
    with pytest.raises(ValueError, match=r"`level` must be 1, 2, or 3, got 4"):
        cfg.for_level(4)


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

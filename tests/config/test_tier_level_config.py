"""Tests for ``TierLevelConfig`` — construction, validation, parsed accessors,
and baked defaults."""

from __future__ import annotations

import pytest

from robotsix_llmio.config.tier import (
    LEVEL1_DEFAULT,
    LEVEL2_DEFAULT,
    LEVEL3_DEFAULT,
    LEVEL4_DEFAULT,
    LEVEL5_DEFAULT,
    TierLevelConfig,
)
from robotsix_llmio.core.identifier import MalformedIdentifierError

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


def test_tier_level_config_openrouter_identifier():
    """An OpenRouter identifier constructs and parses cleanly."""
    cfg = TierLevelConfig(model="openrouter-deepseek/deepseek-v4-pro")
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
        model="openrouter-deepseek/deepseek-v4-flash-latest",
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
        model="openrouter-deepseek/deepseek-v4-flash-latest",
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
    assert LEVEL1_DEFAULT.model == "openrouter-deepseek/deepseek-v4-flash-20260731"
    assert LEVEL1_DEFAULT.provider == "openrouter"
    assert LEVEL1_DEFAULT.model_name == "deepseek/deepseek-v4-flash-20260731"


def test_level2_default():
    assert LEVEL2_DEFAULT.model == "claudeSDK-haiku"
    assert LEVEL2_DEFAULT.provider == "claudeSDK"
    assert LEVEL2_DEFAULT.model_name == "haiku"


def test_level3_default():
    assert LEVEL3_DEFAULT.model == "openrouter-xiaomi/mimo-v2.5-pro"
    assert LEVEL3_DEFAULT.provider == "openrouter"
    assert LEVEL3_DEFAULT.model_name == "xiaomi/mimo-v2.5-pro"


def test_level4_default():
    assert LEVEL4_DEFAULT.model == "claudeSDK-opus"
    assert LEVEL4_DEFAULT.provider == "claudeSDK"
    assert LEVEL4_DEFAULT.model_name == "opus"


def test_level5_default():
    assert LEVEL5_DEFAULT.model == "claudeSDK-claude-fable-5"
    assert LEVEL5_DEFAULT.provider == "claudeSDK"
    assert LEVEL5_DEFAULT.model_name == "claude-fable-5"


# ========================================================================== #
#  TierLevelConfig parsed accessors
# ========================================================================== #


def test_tier_level_config_parsed_accessors_simple():
    """Parsed accessors work for a simple provider-model identifier."""
    cfg = TierLevelConfig(model="claudeSDK-haiku")
    assert cfg.provider == "claudeSDK"
    assert cfg.model_name == "haiku"


def test_tier_level_config_parsed_accessors_openrouter():
    """Parsed accessors work for an OpenRouter identifier whose model name
    contains a slash."""
    cfg = TierLevelConfig(model="openrouter-deepseek/deepseek-v4-pro")
    assert cfg.provider == "openrouter"
    assert cfg.model_name == "deepseek/deepseek-v4-pro"


def test_tier_level_config_parsed_accessors_dash_in_model_name():
    """The model_name portion may itself contain hyphens."""
    cfg = TierLevelConfig(model="claudeSDK-some-model-with-dashes")
    assert cfg.provider == "claudeSDK"
    assert cfg.model_name == "some-model-with-dashes"

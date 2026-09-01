"""Tests for ``TierLevelConfig`` — construction, validation, parsed accessors,
and baked defaults."""

from __future__ import annotations

import pytest

from robotsix_llmio.config.tier import (
    DEFAULT_LEVEL1,
    DEFAULT_LEVEL2,
    DEFAULT_LEVEL3,
    FALLBACK_LEVEL1,
    FALLBACK_LEVEL2,
    FALLBACK_LEVEL3,
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
    cfg = TierLevelConfig(model="openrouter-deepseek/deepseek-v4-pro-0813")
    assert cfg.provider == "openrouter"
    assert cfg.model_name == "deepseek/deepseek-v4-pro-0813"


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


def test_default_slot_levels():
    """The default (Anthropic / Claude SDK) slot: haiku, opus, fable."""
    assert DEFAULT_LEVEL1.model == "claudeSDK-haiku"
    assert DEFAULT_LEVEL2.model == "claudeSDK-opus"
    assert DEFAULT_LEVEL3.model == "claudeSDK-claude-fable-5"
    for tlc in (DEFAULT_LEVEL1, DEFAULT_LEVEL2, DEFAULT_LEVEL3):
        assert tlc.provider == "claudeSDK"
        # No max_tokens on Claude SDK levels — task_budget is advisory only.
        assert tlc.max_tokens is None


def test_fallback_level1():
    assert FALLBACK_LEVEL1.model == "openrouter-deepseek/deepseek-v4-flash-20260731"
    assert FALLBACK_LEVEL1.provider == "openrouter"
    assert FALLBACK_LEVEL1.model_name == "deepseek/deepseek-v4-flash-20260731"
    # Cheap tier prefers a stable *cheap* upstream (DeepInfra), not DeepSeek,
    # whose repriced flash endpoint no longer satisfies the cheap-tier ceiling.
    assert FALLBACK_LEVEL1.provider_kwargs == {"preferred_provider": "DeepInfra"}
    assert FALLBACK_LEVEL1.max_tokens == 16384


def test_fallback_level2_same_flash_with_reasoning_headroom():
    """Level 2 binds the SAME flash snapshot as level 1 — the difference is
    the level>=2 reasoning policy, a larger output cap (reasoning bills
    against it) and an explicit cheap price ceiling (the implicit per-level
    default would apply the capable ceiling)."""
    assert FALLBACK_LEVEL2.model_name == FALLBACK_LEVEL1.model_name
    assert FALLBACK_LEVEL2.max_tokens == 65536
    assert FALLBACK_LEVEL2.provider_kwargs["max_price_prompt"] == 0.10
    assert FALLBACK_LEVEL2.provider_kwargs["max_price_completion"] == 0.20
    assert FALLBACK_LEVEL2.provider_kwargs["preferred_provider"] == "DeepInfra"


def test_fallback_level3():
    assert FALLBACK_LEVEL3.model == "openrouter-deepseek/deepseek-v4-pro-0813"
    assert FALLBACK_LEVEL3.provider == "openrouter"
    assert FALLBACK_LEVEL3.provider_kwargs["preferred_provider"] == "StreamLake"
    assert FALLBACK_LEVEL3.max_tokens == 131072


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
    cfg = TierLevelConfig(model="openrouter-deepseek/deepseek-v4-pro-0813")
    assert cfg.provider == "openrouter"
    assert cfg.model_name == "deepseek/deepseek-v4-pro-0813"


def test_tier_level_config_parsed_accessors_dash_in_model_name():
    """The model_name portion may itself contain hyphens."""
    cfg = TierLevelConfig(model="claudeSDK-some-model-with-dashes")
    assert cfg.provider == "claudeSDK"
    assert cfg.model_name == "some-model-with-dashes"

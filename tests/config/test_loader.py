"""Unit tests for :func:`load_tier_config` — dict-over-baked-defaults merging."""

from __future__ import annotations

import pytest

from robotsix_llmio.config.loader import TierConfigLoadError, load_tier_config
from robotsix_llmio.config.tier import (
    DEFAULT_LEVEL2,
    FALLBACK_LEVEL1,
    FALLBACK_LEVEL3,
    TierConfig,
    TierLevelConfig,
)

# --------------------------------------------------------------------------- #
#  Baked defaults                                                             #
# --------------------------------------------------------------------------- #


def test_none_returns_baked_defaults():
    assert load_tier_config(None) == TierConfig()


def test_empty_dict_returns_baked_defaults():
    assert load_tier_config({}) == TierConfig()


# --------------------------------------------------------------------------- #
#  Per-slot / per-level merging                                               #
# --------------------------------------------------------------------------- #


def test_override_one_level_keeps_rest_baked():
    cfg = load_tier_config({"default": {"level2": {"model": "claudeSDK-sonnet"}}})
    assert cfg.default.level2.model == "claudeSDK-sonnet"
    assert cfg.default.level1.model == "claudeSDK-haiku"
    assert cfg.fallback.level1.model == FALLBACK_LEVEL1.model


def test_field_merge_preserves_baked_siblings():
    """Overriding one field of a level keeps that level's other baked fields."""
    cfg = load_tier_config(
        {"fallback": {"level3": {"model": "openrouter-deepseek/deepseek-v4-pro"}}}
    )
    assert cfg.fallback.level3.model == "openrouter-deepseek/deepseek-v4-pro"
    # provider_kwargs and max_tokens come from the baked FALLBACK_LEVEL3.
    assert cfg.fallback.level3.provider_kwargs == FALLBACK_LEVEL3.provider_kwargs
    assert cfg.fallback.level3.max_tokens == FALLBACK_LEVEL3.max_tokens


def test_failover_merges_field_by_field():
    cfg = load_tier_config({"failover": {"window_seconds": 120}})
    assert cfg.failover.window_seconds == 120.0
    assert cfg.failover.failure_threshold == 3  # baked default preserved


def test_pydantic_models_accepted_as_values():
    cfg = load_tier_config(
        {"default": {"level2": TierLevelConfig(model="claudeSDK-sonnet")}}
    )
    assert cfg.default.level2.model == "claudeSDK-sonnet"


# --------------------------------------------------------------------------- #
#  Failure modes                                                              #
# --------------------------------------------------------------------------- #


def test_legacy_flat_shape_raises_with_migration_hint():
    with pytest.raises(
        TierConfigLoadError, match=r"level1\.\.level5 shape was removed"
    ):
        load_tier_config({"level1": {"model": "claudeSDK-opus"}})


def test_unknown_top_level_key_raises():
    with pytest.raises(TierConfigLoadError, match="Unknown top-level key"):
        load_tier_config({"primary": {}})


def test_unknown_provider_prefix_raises():
    with pytest.raises(TierConfigLoadError, match="Unknown provider prefix"):
        load_tier_config({"default": {"level1": {"model": "nonsense-model"}}})


def test_unknown_level_field_raises():
    with pytest.raises(TierConfigLoadError):
        load_tier_config({"default": {"level1": {"modle": "claudeSDK-haiku"}}})


def test_unmergeable_value_raises():
    with pytest.raises(TierConfigLoadError, match="Cannot merge"):
        load_tier_config({"default": {"level1": "claudeSDK-haiku"}})


def test_unmergeable_slot_raises():
    with pytest.raises(TierConfigLoadError, match="Cannot merge"):
        load_tier_config({"default": ["claudeSDK-haiku"]})


# --------------------------------------------------------------------------- #
#  Re-exports                                                                 #
# --------------------------------------------------------------------------- #


def test_reexport_from_config_package():
    from robotsix_llmio.config import load_tier_config as reexported

    assert reexported is load_tier_config


def test_reexport_from_core_package():
    from robotsix_llmio.core import load_tier_config as reexported

    assert reexported is load_tier_config


def test_default_level2_is_opus():
    """The loader and the schema agree on the baked workhorse binding."""
    assert load_tier_config({}).default.level2 == DEFAULT_LEVEL2


def test_vision_merges_field_by_field():
    cfg = load_tier_config({"vision": {"model": "openrouter-google/gemini-2-flash"}})
    assert cfg.vision.model == "openrouter-google/gemini-2-flash"
    # max_tokens inherits the baked vision default.
    assert cfg.vision.max_tokens == 8192

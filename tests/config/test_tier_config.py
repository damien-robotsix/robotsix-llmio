"""Unit tests for :class:`TierConfig` — two provider slots, three levels."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from robotsix_llmio.config.tier import (
    DEFAULT_LEVEL1,
    DEFAULT_LEVEL2,
    DEFAULT_LEVEL3,
    FALLBACK_LEVEL1,
    FALLBACK_LEVEL2,
    FALLBACK_LEVEL3,
    FailoverConfig,
    ProviderSlotConfig,
    TierConfig,
    TierLevelConfig,
)

# --------------------------------------------------------------------------- #
#  Construction & baked defaults                                              #
# --------------------------------------------------------------------------- #


def test_no_args_yields_baked_slots():
    cfg = TierConfig()
    assert cfg.default.level1.model == DEFAULT_LEVEL1.model == "claudeSDK-haiku"
    assert cfg.default.level2.model == DEFAULT_LEVEL2.model == "claudeSDK-opus"
    assert (
        cfg.default.level3.model == DEFAULT_LEVEL3.model == "claudeSDK-claude-fable-5"
    )
    assert cfg.fallback.level1.model == FALLBACK_LEVEL1.model
    assert cfg.fallback.level2.model == FALLBACK_LEVEL2.model
    assert cfg.fallback.level3.model == FALLBACK_LEVEL3.model
    assert cfg.failover == FailoverConfig()


def test_fallback_slot_is_openrouter_deepseek():
    cfg = TierConfig()
    assert cfg.fallback.level1.provider == "openrouter"
    # Operator design (re-confirmed 2026-09-01): levels 1 and 2 bind the SAME
    # flash snapshot (level 2 adds xhigh reasoning + a larger output cap);
    # level 3 is pro. The flash-loop risk at L2 is a recorded, accepted
    # trade-off — see the FALLBACK_LEVEL2 comment in config/tier.py.
    assert cfg.fallback.level1.model_name == "deepseek/deepseek-v4-flash-20260731"
    assert cfg.fallback.level2.model_name == "deepseek/deepseek-v4-flash-20260731"
    assert cfg.fallback.level3.model_name == "deepseek/deepseek-v4-pro-0813"
    assert cfg.fallback.level2.max_tokens > cfg.fallback.level1.max_tokens


def test_default_factories_do_not_alias():
    a, b = TierConfig(), TierConfig()
    a.default.level1.provider_kwargs["x"] = 1
    assert "x" not in b.default.level1.provider_kwargs
    assert "x" not in DEFAULT_LEVEL1.provider_kwargs


def test_partial_override_keeps_other_slots_baked():
    cfg = TierConfig(
        default=ProviderSlotConfig(
            level1=TierLevelConfig(model="claudeSDK-sonnet"),
            level2=DEFAULT_LEVEL2,
            level3=DEFAULT_LEVEL3,
        )
    )
    assert cfg.default.level1.model == "claudeSDK-sonnet"
    assert cfg.fallback.level3.model == FALLBACK_LEVEL3.model


def test_model_validate_nested_dict():
    cfg = TierConfig.model_validate(
        {
            "default": {
                "level1": {"model": "claudeSDK-haiku"},
                "level2": {"model": "claudeSDK-opus"},
                "level3": {"model": "claudeSDK-claude-fable-5"},
            },
            "failover": {"failure_threshold": 5, "window_seconds": 60},
        }
    )
    assert cfg.failover.failure_threshold == 5
    assert cfg.failover.window_seconds == 60.0


def test_model_dump_round_trip():
    cfg = TierConfig()
    assert TierConfig.model_validate(cfg.model_dump()) == cfg


def test_legacy_flat_level_keys_are_rejected():
    """The pre-rework flat ``level1..level5`` shape must fail loudly, not
    silently validate to baked defaults."""
    with pytest.raises(ValidationError):
        TierConfig.model_validate({"level1": {"model": "claudeSDK-opus"}})


def test_unknown_slot_key_rejected():
    with pytest.raises(ValidationError):
        TierConfig.model_validate({"primary": {}})


# --------------------------------------------------------------------------- #
#  for_level / slot resolution                                                #
# --------------------------------------------------------------------------- #


def test_for_level_explicit_slots():
    cfg = TierConfig()
    assert cfg.for_level(1, slot="default").model == "claudeSDK-haiku"
    assert cfg.for_level(2, slot="default").model == "claudeSDK-opus"
    assert cfg.for_level(3, slot="default").model == "claudeSDK-claude-fable-5"
    assert cfg.for_level(1, slot="fallback").provider == "openrouter"
    assert (
        cfg.for_level(3, slot="fallback").model_name == "deepseek/deepseek-v4-pro-0813"
    )


@pytest.mark.parametrize("bad_level", [0, 4, 5, -1])
def test_for_level_out_of_range_raises(bad_level: int):
    with pytest.raises(ValueError, match="must be 1, 2, or 3"):
        TierConfig().for_level(bad_level, slot="default")


def test_slot_accessor_rejects_unknown_name():
    with pytest.raises(ValueError, match="'default' or 'fallback'"):
        TierConfig().slot("primary")  # type: ignore[arg-type]


def test_for_level_default_slot_is_tracker_driven():
    """With no explicit slot, resolution follows the failover tracker (the
    tracker starts on ``default``; the armed path is covered in
    tests/core/test_core_failover.py)."""
    cfg = TierConfig()
    assert cfg.for_level(2).model == cfg.for_level(2, slot="default").model


# --------------------------------------------------------------------------- #
#  FailoverConfig bounds                                                      #
# --------------------------------------------------------------------------- #


def test_failover_defaults_are_three_failures_fifteen_minutes():
    cfg = FailoverConfig()
    assert cfg.failure_threshold == 3
    assert cfg.window_seconds == 900.0


@pytest.mark.parametrize(
    "kwargs",
    [{"failure_threshold": 0}, {"window_seconds": 0}, {"window_seconds": -5}],
)
def test_failover_bounds_enforced(kwargs: dict):
    with pytest.raises(ValidationError):
        FailoverConfig(**kwargs)


# --------------------------------------------------------------------------- #
#  Vision binding                                                             #
# --------------------------------------------------------------------------- #


def test_vision_binding_baked_default():
    cfg = TierConfig()
    assert cfg.vision.model == "openrouter-deepseek/deepseek-v4-flash-vision-exp"
    assert cfg.vision.max_tokens == 8192
    assert cfg.vision.provider == "openrouter"


def test_vision_binding_override():
    cfg = TierConfig(vision=TierLevelConfig(model="openrouter-google/gemini-2-flash"))
    assert cfg.vision.model == "openrouter-google/gemini-2-flash"

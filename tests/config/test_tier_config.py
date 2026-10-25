"""Unit tests for :class:`TierConfig` — ordered provider-slot chain, three levels."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from robotsix_llmio.config.tier import (
    ANTHROPIC_LEVEL1,
    ANTHROPIC_LEVEL2,
    ANTHROPIC_LEVEL3,
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
    anthropic, sdk, openrouter = cfg.chain
    assert anthropic.level1.model == ANTHROPIC_LEVEL1.model
    assert anthropic.level2.model == ANTHROPIC_LEVEL2.model
    assert anthropic.level3.model == ANTHROPIC_LEVEL3.model
    assert sdk.level1.model == DEFAULT_LEVEL1.model == "claudeSDK-haiku"
    assert sdk.level2.model == DEFAULT_LEVEL2.model == "claudeSDK-opus"
    assert sdk.level3.model == DEFAULT_LEVEL3.model == "claudeSDK-claude-fable-5"
    assert openrouter.level1.model == FALLBACK_LEVEL1.model
    assert openrouter.level2.model == FALLBACK_LEVEL2.model
    assert openrouter.level3.model == FALLBACK_LEVEL3.model
    assert cfg.failover == FailoverConfig()


@pytest.mark.parametrize("level", [1, 2, 3])
def test_baked_chain_order_is_anthropic_claude_sdk_openrouter(level: int):
    cfg = TierConfig()
    providers = [slot.for_level(level).provider for slot in cfg.chain]
    assert providers == ["anthropic", "claudeSDK", "openrouter"]


def test_anthropic_slot_mirrors_claude_sdk_bindings():
    cfg = TierConfig()
    anthropic, sdk = cfg.chain[0], cfg.chain[1]
    assert anthropic.level1.model_name == "claude-haiku-5-5"
    assert sdk.level1.model_name == "haiku"
    assert anthropic.level2.model_name == "claude-opus-5-5"
    assert sdk.level2.model_name == "opus"
    assert anthropic.level3.model_name == sdk.level3.model_name == "claude-fable-5"


def test_fallback_slot_is_openrouter_deepseek():
    cfg = TierConfig()
    last = cfg.chain[-1]
    assert last.level1.provider == "openrouter"
    # Operator design (re-confirmed 2026-09-01): levels 1 and 2 bind the SAME
    # flash snapshot (level 2 adds xhigh reasoning + a larger output cap);
    # level 3 is pro. The flash-loop risk at L2 is a recorded, accepted
    # trade-off — see the FALLBACK_LEVEL2 comment in config/tier.py.
    assert last.level1.model_name == "deepseek/deepseek-v4-flash-20260731"
    assert last.level2.model_name == "deepseek/deepseek-v4-flash-20260731"
    assert last.level3.model_name == "deepseek/deepseek-v4-pro-0813"
    assert last.level2.max_tokens > last.level1.max_tokens


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
    assert cfg.for_level(1, slot="default").model == "anthropic-claude-haiku-5-5"
    assert cfg.for_level(2, slot="default").model == "anthropic-claude-opus-5-5"
    assert cfg.for_level(3, slot="default").model == "anthropic-claude-fable-5"
    assert cfg.for_level(1, slot="fallback").model == "claudeSDK-haiku"
    assert cfg.for_level(3, slot="fallback").model == "claudeSDK-claude-fable-5"
    assert cfg.for_level(1, slot=2).provider == "openrouter"
    assert cfg.for_level(3, slot=2).model_name == "deepseek/deepseek-v4-pro-0813"


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
#  Ordered N-slot chain + legacy two-slot migration                           #
# --------------------------------------------------------------------------- #


def _slot(l1: str, l2: str, l3: str) -> dict[str, dict[str, str]]:
    return {"level1": {"model": l1}, "level2": {"model": l2}, "level3": {"model": l3}}


def test_baked_chain_is_default_then_fallback():
    cfg = TierConfig()
    assert len(cfg.chain) == 3
    assert cfg.chain[0] == cfg.default == cfg.slot(0) == cfg.slot("default")
    assert cfg.chain[1] == cfg.fallback == cfg.slot(1) == cfg.slot("fallback")
    assert cfg.chain[2] == cfg.slot(2)


def test_three_slot_chain_resolves_every_position():
    cfg = TierConfig.model_validate(
        {
            "chain": [
                _slot("claudeSDK-haiku", "claudeSDK-opus", "claudeSDK-claude-fable-5"),
                _slot("claudeSDK-sonnet", "claudeSDK-sonnet", "claudeSDK-opus"),
                _slot(
                    FALLBACK_LEVEL1.model, FALLBACK_LEVEL2.model, FALLBACK_LEVEL3.model
                ),
            ]
        }
    )
    assert len(cfg.chain) == 3
    assert cfg.for_level(1, slot=0).model == "claudeSDK-haiku"
    assert cfg.for_level(3, slot=1).model == "claudeSDK-opus"
    assert cfg.for_level(2, slot=2).model == FALLBACK_LEVEL2.model
    assert cfg.slot(2).level3.provider == "openrouter"
    assert TierConfig.model_validate(cfg.model_dump()) == cfg


def test_legacy_keys_migrate_onto_chain():
    cfg = TierConfig.model_validate(
        {
            "default": _slot("claudeSDK-sonnet", "claudeSDK-opus", "claudeSDK-opus"),
            "fallback": _slot(
                FALLBACK_LEVEL3.model, FALLBACK_LEVEL3.model, FALLBACK_LEVEL3.model
            ),
        }
    )
    assert len(cfg.chain) == 2
    assert cfg.chain[0].level1.model == "claudeSDK-sonnet"
    assert cfg.chain[1].level1.model == FALLBACK_LEVEL3.model


def test_legacy_fallback_only_keeps_baked_default():
    cfg = TierConfig.model_validate(
        {"fallback": _slot("claudeSDK-haiku", "claudeSDK-haiku", "claudeSDK-haiku")}
    )
    assert cfg.chain[0].level2.model == DEFAULT_LEVEL2.model
    assert cfg.chain[1].level2.model == "claudeSDK-haiku"


def test_legacy_keys_mixed_with_chain_rejected():
    with pytest.raises(ValidationError, match="legacy slot key"):
        TierConfig.model_validate(
            {
                "default": _slot("claudeSDK-haiku", "claudeSDK-opus", "claudeSDK-opus"),
                "chain": [_slot("claudeSDK-haiku", "claudeSDK-opus", "claudeSDK-opus")],
            }
        )


def test_empty_chain_rejected():
    with pytest.raises(ValidationError):
        TierConfig.model_validate({"chain": []})


@pytest.mark.parametrize("bad_slot", [3, -1, True])
def test_slot_out_of_chain_rejected(bad_slot: int):
    with pytest.raises(ValueError, match="chain index"):
        TierConfig().slot(bad_slot)


def test_legacy_fallback_alias_rejected_on_single_slot_chain():
    cfg = TierConfig.model_validate(
        {"chain": [_slot("claudeSDK-haiku", "claudeSDK-opus", "claudeSDK-opus")]}
    )
    assert cfg.default.level1.model == "claudeSDK-haiku"
    with pytest.raises(ValueError, match="chain index"):
        cfg.slot("fallback")


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


def test_failover_backoff_defaults():
    cfg = FailoverConfig()
    assert cfg.reattempt_backoff_base == 2.0
    assert cfg.reattempt_backoff_cap_seconds == 14400.0
    assert cfg.reattempt_backoff_jitter == 0.5


@pytest.mark.parametrize(
    "kwargs",
    [
        {"reattempt_backoff_base": 1.0},  # must be > 1
        {"reattempt_backoff_base": 0.5},
        {"reattempt_backoff_cap_seconds": 0},  # must be > 0
        {"reattempt_backoff_jitter": -0.1},  # must be in [0, 1]
        {"reattempt_backoff_jitter": 1.1},
        {"window_seconds": 20000.0},  # cap (default 14400) must be >= window
    ],
)
def test_failover_backoff_bounds_enforced(kwargs: dict):
    with pytest.raises(ValidationError):
        FailoverConfig(**kwargs)


def test_failover_cost_guard_defaults_disabled():
    """The per-request cost guard is opt-in — both knobs default to None."""
    cfg = FailoverConfig()
    assert cfg.max_request_cost_usd is None
    assert cfg.cost_ratio_threshold is None


def test_failover_cost_guard_is_configurable():
    cfg = FailoverConfig(max_request_cost_usd=0.25, cost_ratio_threshold=1.5)
    assert cfg.max_request_cost_usd == 0.25
    assert cfg.cost_ratio_threshold == 1.5


@pytest.mark.parametrize(
    "kwargs",
    [
        {"max_request_cost_usd": 0},  # must be > 0
        {"max_request_cost_usd": -1.0},
        {"cost_ratio_threshold": 0},  # must be > 0
        {"cost_ratio_threshold": -0.5},
    ],
)
def test_failover_cost_guard_bounds_enforced(kwargs: dict):
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

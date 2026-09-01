"""Tests for the level-based factory entry points (``core.factory``).

These cover :func:`default_tier_config`, :func:`get_provider_for_level`, and
:func:`build_agent_for_level`.  ``get_provider_for_identifier`` is patched so
the tests never need a real backend (and so the ``claude_sdk`` extra is never
actually required to instantiate a default-slot provider).
"""

from __future__ import annotations

from typing import Any

import pytest

from robotsix_llmio.config.tier import (
    FALLBACK_LEVEL1,
    FALLBACK_LEVEL2,
    FALLBACK_LEVEL3,
    ProviderSlotConfig,
    TierConfig,
    TierLevelConfig,
)
from robotsix_llmio.core import factory
from robotsix_llmio.core.factory import (
    build_agent_for_level,
    default_tier_config,
    get_provider_for_level,
)


class _FakeProvider:
    """Captures the kwargs passed to ``build_agent`` for assertion."""

    def __init__(self) -> None:
        self.build_agent_calls: list[dict[str, Any]] = []

    def build_agent(self, **kwargs: Any) -> str:
        self.build_agent_calls.append(kwargs)
        return "agent-handle"


def _patch_factory(
    monkeypatch: pytest.MonkeyPatch,
) -> tuple[list[tuple[str, dict[str, Any]]], _FakeProvider]:
    """Patch ``get_provider_for_identifier`` to record (identifier, kwargs) and
    return a shared ``_FakeProvider``.  Returns the call log and the provider.
    """
    calls: list[tuple[str, dict[str, Any]]] = []
    provider = _FakeProvider()

    def _fake_get(identifier: str, **kwargs: Any) -> _FakeProvider:
        calls.append((identifier, kwargs))
        return provider

    monkeypatch.setattr(factory, "get_provider_for_identifier", _fake_get)
    return calls, provider


def _slot_with_level1(level1: TierLevelConfig) -> ProviderSlotConfig:
    """A slot pinning *level1*, filling levels 2-3 with claudeSDK models."""
    return ProviderSlotConfig(
        level1=level1,
        level2=TierLevelConfig(model="claudeSDK-opus"),
        level3=TierLevelConfig(model="claudeSDK-claude-fable-5"),
    )


# -- default_tier_config ----------------------------------------------------


def test_default_tier_config_bakes_per_level_defaults() -> None:
    cfg = default_tier_config()

    # Default slot: Anthropic via the Claude SDK.
    assert cfg.for_level(1, slot="default").model == "claudeSDK-haiku"
    assert cfg.for_level(2, slot="default").model == "claudeSDK-opus"
    assert cfg.for_level(3, slot="default").model == "claudeSDK-claude-fable-5"

    # Fallback slot: DeepSeek via OpenRouter (flash / flash / pro).
    assert (
        cfg.for_level(1, slot="fallback").model
        == "openrouter-deepseek/deepseek-v4-flash-20260731"
    )
    assert (
        cfg.for_level(2, slot="fallback").model
        == "openrouter-deepseek/deepseek-v4-flash-20260731"
    )
    assert (
        cfg.for_level(3, slot="fallback").model
        == "openrouter-deepseek/deepseek-v4-pro-0813"
    )


def test_default_tier_config_active_slot_is_default() -> None:
    """With no failover armed, slot-less resolution follows the default slot."""
    cfg = default_tier_config()

    assert cfg.for_level(1).model == "claudeSDK-haiku"
    assert cfg.for_level(2).model == "claudeSDK-opus"
    assert cfg.for_level(3).model == "claudeSDK-claude-fable-5"


# -- get_provider_for_level -------------------------------------------------


def test_get_provider_for_level_resolves_per_level_identifier(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls, provider = _patch_factory(monkeypatch)

    assert get_provider_for_level(1) is provider
    assert get_provider_for_level(2) is provider

    # Both default-slot levels resolve claudeSDK identifiers.
    assert calls[0][0] == "claudeSDK-haiku"
    assert calls[1][0] == "claudeSDK-opus"


def test_get_provider_for_level_merges_provider_kwargs(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls, _ = _patch_factory(monkeypatch)

    tier_config = TierConfig(
        default=_slot_with_level1(
            TierLevelConfig(
                model="openrouter-deepseek/deepseek-v4-flash-latest",
                provider_kwargs={"base_url": "https://proxy", "api_key": "from-tier"},
            )
        ),
    )

    get_provider_for_level(1, tier_config=tier_config, api_key="caller-wins")

    identifier, kwargs = calls[0]
    assert identifier == "openrouter-deepseek/deepseek-v4-flash-latest"
    # tier provider_kwargs merged with caller kwargs; caller wins on conflict.
    assert kwargs == {"base_url": "https://proxy", "api_key": "caller-wins"}


def test_get_provider_for_level_invalid_level_raises(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_factory(monkeypatch)
    with pytest.raises(ValueError):
        get_provider_for_level(4)


# -- build_agent_for_level --------------------------------------------------


def test_build_agent_for_level_default_level1(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls, provider = _patch_factory(monkeypatch)

    handle = build_agent_for_level(
        1, system_prompt="cheap task", output_type=str, name="lvl1"
    )

    assert handle == "agent-handle"
    # ClaudeSDK provider resolved from the default slot's level-1 identifier.
    assert calls[0][0] == "claudeSDK-haiku"
    # build_agent got level= and the bare level-1 model name.
    call = provider.build_agent_calls[0]
    assert call["level"] == 1
    assert call["model"] == "haiku"
    assert call["system_prompt"] == "cheap task"
    assert call["name"] == "lvl1"


def test_build_agent_for_level_default_level2_opus(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls, provider = _patch_factory(monkeypatch)

    build_agent_for_level(2, system_prompt="workhorse", tools=[], output_type=str)

    # ClaudeSDK provider resolved from the level-2 identifier.
    assert calls[0][0] == "claudeSDK-opus"
    call = provider.build_agent_calls[0]
    assert call["level"] == 2
    assert call["model"] == "opus"


def test_build_agent_for_level_default_level3(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls, provider = _patch_factory(monkeypatch)

    build_agent_for_level(3, system_prompt="frontier task", output_type=str)

    # ClaudeSDK provider resolved from the level-3 identifier.
    assert calls[0][0] == "claudeSDK-claude-fable-5"
    call = provider.build_agent_calls[0]
    assert call["level"] == 3
    assert call["model"] == "claude-fable-5"


def test_build_agent_for_level_model_override_keeps_provider(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls, provider = _patch_factory(monkeypatch)

    build_agent_for_level(1, model="sonnet", system_prompt="x")

    # Provider still resolved from the level's identifier (claudeSDK).
    assert calls[0][0] == "claudeSDK-haiku"
    # Only the model name is overridden.
    assert provider.build_agent_calls[0]["model"] == "sonnet"


def test_build_agent_for_level_custom_tier_config(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls, provider = _patch_factory(monkeypatch)

    tier_config = TierConfig(
        default=_slot_with_level1(TierLevelConfig(model="claudeSDK-sonnet")),
    )

    build_agent_for_level(1, tier_config=tier_config, system_prompt="x")

    # Custom tier config overrides the baked default for level 1.
    assert calls[0][0] == "claudeSDK-sonnet"
    assert provider.build_agent_calls[0]["model"] == "sonnet"


def test_build_agent_for_level_provider_kwargs_forwarded(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls, _ = _patch_factory(monkeypatch)

    # An OpenRouter binding in the default slot: its baked max_tokens must be
    # forwarded alongside the explicit provider_kwargs.
    tier_config = TierConfig(
        default=ProviderSlotConfig(
            level1=FALLBACK_LEVEL1,
            level2=FALLBACK_LEVEL2,
            level3=FALLBACK_LEVEL3,
        ),
    )

    build_agent_for_level(
        1,
        tier_config=tier_config,
        provider_kwargs={"api_key": "explicit"},
        system_prompt="x",
    )

    assert calls[0][1] == {"api_key": "explicit", "max_tokens": 16384}


def test_build_agent_for_level_no_max_tokens_on_claude_defaults(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls, _ = _patch_factory(monkeypatch)

    build_agent_for_level(1, provider_kwargs={"api_key": "explicit"}, system_prompt="x")

    # The baked claudeSDK levels carry no max_tokens, so none is forwarded.
    assert calls[0][1] == {"api_key": "explicit"}


def test_build_agent_for_level_tier_provider_kwargs_forwarded(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls, _ = _patch_factory(monkeypatch)

    tier_config = TierConfig(
        default=_slot_with_level1(
            TierLevelConfig(
                model="openrouter-deepseek/deepseek-v4-flash-latest",
                provider_kwargs={"base_url": "https://proxy"},
            )
        ),
    )

    build_agent_for_level(1, tier_config=tier_config, system_prompt="x")

    # With no explicit provider_kwargs, the tier level's are forwarded.
    assert calls[0][1] == {"base_url": "https://proxy"}


def test_build_agent_for_level_invalid_level_raises(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_factory(monkeypatch)
    with pytest.raises(ValueError):
        build_agent_for_level(4, system_prompt="x")


# -- exports ----------------------------------------------------------------


def test_level_helpers_exported_from_core() -> None:
    import robotsix_llmio.core as core

    assert core.build_agent_for_level is build_agent_for_level
    assert core.get_provider_for_level is get_provider_for_level
    assert core.default_tier_config is default_tier_config


def test_level_helpers_exported_top_level() -> None:
    import robotsix_llmio as llmio

    assert llmio.build_agent_for_level is build_agent_for_level
    assert llmio.get_provider_for_level is get_provider_for_level
    assert llmio.default_tier_config is default_tier_config


def test_get_provider_for_level_explicit_slot(monkeypatch):
    """``slot="fallback"`` resolves the fallback binding even while the
    tracker's active slot is ``default`` — the seam a consumer's failover
    loop uses for its cross-slot attempt."""
    from unittest.mock import MagicMock

    from robotsix_llmio.core import factory as core_factory

    mock = MagicMock()
    monkeypatch.setattr(core_factory, "get_provider_for_identifier", mock)

    core_factory.get_provider_for_level(2, slot="fallback")

    identifier = mock.call_args.args[0]
    assert identifier == FALLBACK_LEVEL2.model


def test_create_model_explicit_slot(monkeypatch):
    from unittest.mock import MagicMock

    from robotsix_llmio.config import factory as config_factory

    mock = MagicMock()
    monkeypatch.setattr(config_factory, "get_provider_for_identifier", mock)

    config_factory.create_model(level=3, slot="fallback")

    assert mock.call_args.args[0] == FALLBACK_LEVEL3.model

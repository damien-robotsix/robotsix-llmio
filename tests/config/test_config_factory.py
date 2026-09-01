"""Tests for :mod:`robotsix_llmio.config.factory` — the consumer-facing
``create_model`` entry-point."""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock

import pytest

from robotsix_llmio.config.factory import create_model
from robotsix_llmio.config.tier import (
    DEFAULT_LEVEL2,
    DEFAULT_LEVEL3,
    FALLBACK_LEVEL1,
    FALLBACK_LEVEL2,
    FALLBACK_LEVEL3,
    ProviderSlotConfig,
    TierConfig,
    TierLevelConfig,
)


class TestCreateModelValidation:
    """Input validation before delegation to the provider factory."""

    def test_invalid_level_raises_valueerror(self):
        for bad_level in (0, 4, 5, -1, 99):
            with pytest.raises(ValueError) as excinfo:
                create_model(level=bad_level)
            message = str(excinfo.value)
            assert "level" in message
            assert str(bad_level) in message


class TestCreateModelHappyPath:
    """Valid calls resolve the active (default) slot's binding for the level."""

    @pytest.mark.parametrize(
        ("level", "identifier"),
        [
            (1, "claudeSDK-haiku"),
            (2, "claudeSDK-opus"),
            (3, "claudeSDK-claude-fable-5"),
        ],
    )
    def test_default_slot_resolution(
        self, level: int, identifier: str, mock_get_provider_for_identifier: MagicMock
    ):
        """With no ``tier_config`` and no failover armed, each level resolves
        the baked default (Claude SDK) slot. No max_tokens kwarg: the Claude
        SDK levels carry none, because the SDK has no per-response cap and
        the value could only become an advisory task_budget (see tier.py)."""
        result = create_model(level=level)
        mock_get_provider_for_identifier.assert_called_once_with(identifier)
        assert result is mock_get_provider_for_identifier.return_value

    def test_fallback_slot_resolves_when_failover_armed(
        self, mock_get_provider_for_identifier: MagicMock
    ):
        """Once the failover tracker arms, the same level resolves the
        fallback (OpenRouter) slot — max_tokens IS forwarded there (a real
        per-response cap)."""
        from robotsix_llmio.core.failover import get_failover_tracker
        from robotsix_llmio.exceptions import ProviderExhaustedError

        get_failover_tracker().record_failure(
            "default", ProviderExhaustedError("out of credits")
        )

        create_model(level=3)
        mock_get_provider_for_identifier.assert_called_once_with(
            FALLBACK_LEVEL3.model,
            **FALLBACK_LEVEL3.provider_kwargs,
            max_tokens=FALLBACK_LEVEL3.max_tokens,
        )

    def test_provider_kwargs_override_tier_config_defaults(
        self, mock_get_provider_for_identifier: MagicMock
    ):
        """Explicit ``provider_kwargs`` passed to ``create_model`` override
        those from the tier config."""
        cfg = TierConfig(
            default=ProviderSlotConfig(
                level1=TierLevelConfig(
                    model="openrouter-deepseek/deepseek-v4-flash-latest",
                    provider_kwargs={
                        "base_url": "https://from-tier.example.com",
                        "api_key": "tier-key",
                    },
                ),
                level2=DEFAULT_LEVEL2,
                level3=DEFAULT_LEVEL3,
            )
        )

        create_model(level=1, tier_config=cfg, api_key="explicit-key")
        mock_get_provider_for_identifier.assert_called_once_with(
            "openrouter-deepseek/deepseek-v4-flash-latest",
            base_url="https://from-tier.example.com",
            api_key="explicit-key",
        )


class TestCreateModelMaxTokens:
    """``max_tokens`` is forwarded only when the resolved level sets one."""

    @pytest.mark.parametrize("tlc", [FALLBACK_LEVEL1, FALLBACK_LEVEL2, FALLBACK_LEVEL3])
    def test_openrouter_levels_forward_max_tokens(
        self, tlc: TierLevelConfig, mock_get_provider_for_identifier: MagicMock
    ):
        cfg = TierConfig(default=ProviderSlotConfig(level1=tlc, level2=tlc, level3=tlc))
        create_model(level=1, tier_config=cfg)

        expected_kwargs: dict[str, Any] = {**tlc.provider_kwargs}
        expected_kwargs.setdefault("max_tokens", tlc.max_tokens)
        mock_get_provider_for_identifier.assert_called_once_with(
            tlc.model, **expected_kwargs
        )

    def test_explicit_tier_config_overrides_defaults(
        self, mock_get_provider_for_identifier: MagicMock
    ):
        """When a ``tier_config`` is supplied, its identifier is used instead
        of the baked default."""
        cfg = TierConfig(
            default=ProviderSlotConfig(
                level1=TierLevelConfig(model="claudeSDK-sonnet"),
                level2=DEFAULT_LEVEL2,
                level3=DEFAULT_LEVEL3,
            )
        )
        create_model(level=1, tier_config=cfg)
        mock_get_provider_for_identifier.assert_called_once_with("claudeSDK-sonnet")

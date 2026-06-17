"""Tests for :mod:`robotsix_llmio.config.factory` — the consumer-facing
``create_model`` entry-point."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from robotsix_llmio.config.factory import create_model
from robotsix_llmio.core.provider import LLMProvider


class TestCreateModelValidation:
    """Input validation before delegation to ``get_provider``."""

    def test_invalid_transport_raises_valueerror(self):
        with pytest.raises(ValueError) as excinfo:
            create_model(level=1, transport="unknown-transport")
        message = str(excinfo.value)
        assert "unknown-transport" in message
        assert "claude-sdk" in message
        assert "openrouter[deepseek]" in message

    def test_invalid_level_raises_valueerror(self):
        for bad_level in (0, 4, -1, 99):
            with pytest.raises(ValueError) as excinfo:
                create_model(level=bad_level)
            message = str(excinfo.value)
            assert "level" in message
            assert str(bad_level) in message


class TestCreateModelHappyPath:
    """Valid calls delegate to ``get_provider`` with the resolved provider name."""

    @pytest.fixture
    def mock_get_provider(self, monkeypatch: pytest.MonkeyPatch) -> MagicMock:
        mock = MagicMock(return_value=MagicMock(spec=LLMProvider))
        monkeypatch.setattr("robotsix_llmio.config.factory.get_provider", mock)
        return mock

    # -- Transport-based tests (explicit transport, legacy path) --------------

    def test_claude_sdk_level_1(self, mock_get_provider: MagicMock):
        result = create_model(level=1, transport="claude-sdk")
        mock_get_provider.assert_called_once_with(provider="claude-sdk")
        assert result is mock_get_provider.return_value

    def test_claude_sdk_level_2(self, mock_get_provider: MagicMock):
        result = create_model(level=2, transport="claude-sdk")
        mock_get_provider.assert_called_once_with(provider="claude-sdk")
        assert result is mock_get_provider.return_value

    def test_claude_sdk_level_3(self, mock_get_provider: MagicMock):
        result = create_model(level=3, transport="claude-sdk")
        mock_get_provider.assert_called_once_with(provider="claude-sdk")
        assert result is mock_get_provider.return_value

    def test_openrouter_deepseek_level_1(self, mock_get_provider: MagicMock):
        result = create_model(level=1, transport="openrouter[deepseek]")
        mock_get_provider.assert_called_once_with(provider="openrouter-deepseek")
        assert result is mock_get_provider.return_value

    def test_openrouter_deepseek_level_2(self, mock_get_provider: MagicMock):
        result = create_model(level=2, transport="openrouter[deepseek]")
        mock_get_provider.assert_called_once_with(provider="openrouter-deepseek")
        assert result is mock_get_provider.return_value

    def test_openrouter_deepseek_level_3(self, mock_get_provider: MagicMock):
        result = create_model(level=3, transport="openrouter[deepseek]")
        mock_get_provider.assert_called_once_with(provider="openrouter-deepseek")
        assert result is mock_get_provider.return_value

    def test_provider_kwargs_are_forwarded(self, mock_get_provider: MagicMock):
        create_model(
            level=2,
            transport="openrouter[deepseek]",
            api_key="my-key",
            base_url="https://proxy.example.com",
        )
        mock_get_provider.assert_called_once_with(
            provider="openrouter-deepseek",
            api_key="my-key",
            base_url="https://proxy.example.com",
        )

    # -- Level-driven resolution (no transport) -------------------------------

    def test_level_1_no_transport_resolves_from_tier_config(
        self, mock_get_provider: MagicMock
    ):
        """``create_model(level=1)`` resolves provider from LEVEL1_DEFAULT."""
        result = create_model(level=1)
        mock_get_provider.assert_called_once_with(
            provider="openrouter-deepseek",
        )
        assert result is mock_get_provider.return_value

    def test_level_2_no_transport_resolves_from_tier_config(
        self, mock_get_provider: MagicMock
    ):
        """``create_model(level=2)`` resolves provider from LEVEL2_DEFAULT."""
        result = create_model(level=2)
        mock_get_provider.assert_called_once_with(
            provider="openrouter-deepseek",
        )
        assert result is mock_get_provider.return_value

    def test_level_3_no_transport_resolves_from_tier_config(
        self, mock_get_provider: MagicMock
    ):
        """``create_model(level=3)`` resolves provider from LEVEL3_DEFAULT
        (``"claude-sdk"``)."""
        result = create_model(level=3)
        mock_get_provider.assert_called_once_with(
            provider="claude-sdk",
        )
        assert result is mock_get_provider.return_value

    # -- Transport override of level-based provider ---------------------------

    def test_transport_override_of_level_3(self, mock_get_provider: MagicMock):
        """``create_model(level=3, transport="openrouter[deepseek]")`` uses
        OpenRouter despite level 3 defaulting to Claude SDK."""
        result = create_model(level=3, transport="openrouter[deepseek]")
        mock_get_provider.assert_called_once_with(
            provider="openrouter-deepseek",
        )
        assert result is mock_get_provider.return_value

    # -- provider_kwargs merging ----------------------------------------------

    def test_provider_kwargs_override_tier_config_defaults(
        self, mock_get_provider: MagicMock
    ):
        """Explicit ``provider_kwargs`` passed to ``create_model`` override
        those from the tier config."""
        from robotsix_llmio.config.tier import (
            LEVEL2_DEFAULT,
            LEVEL3_DEFAULT,
            TierConfig,
            TierLevelConfig,
        )

        cfg = TierConfig(
            level1=TierLevelConfig(
                transport="openrouter[deepseek]",
                model="deepseek/deepseek-v4-flash",
                provider_kwargs={
                    "base_url": "https://from-tier.example.com",
                    "api_key": "tier-key",
                },
            ),
            level2=LEVEL2_DEFAULT,
            level3=LEVEL3_DEFAULT,
        )

        create_model(
            level=1,
            tier_config=cfg,
            api_key="explicit-key",
        )
        mock_get_provider.assert_called_once_with(
            provider="openrouter-deepseek",
            base_url="https://from-tier.example.com",
            api_key="explicit-key",
        )


class TestCreateModelDefaultFallback:
    """``create_model`` falls back to the baked level defaults only when no
    user-supplied ``tier_config`` is present."""

    @pytest.fixture
    def mock_get_provider(self, monkeypatch: pytest.MonkeyPatch) -> MagicMock:
        mock = MagicMock(return_value=MagicMock(spec=LLMProvider))
        monkeypatch.setattr("robotsix_llmio.config.factory.get_provider", mock)
        return mock

    @pytest.mark.parametrize("level", [1, 2, 3])
    def test_no_tier_config_uses_baked_level_default(
        self, level: int, mock_get_provider: MagicMock
    ):
        """With no ``tier_config``, the provider resolves from the matching
        ``LEVEL{1,2,3}_DEFAULT`` constant."""
        from robotsix_llmio.config.tier import (
            LEVEL1_DEFAULT,
            LEVEL2_DEFAULT,
            LEVEL3_DEFAULT,
        )

        expected = {1: LEVEL1_DEFAULT, 2: LEVEL2_DEFAULT, 3: LEVEL3_DEFAULT}[level]

        create_model(level=level)

        mock_get_provider.assert_called_once_with(provider=expected.provider)

    def test_explicit_tier_config_overrides_defaults(
        self, mock_get_provider: MagicMock
    ):
        """When a ``tier_config`` is supplied, its provider is used instead of
        the baked default — defaults only apply when no config exists."""
        from robotsix_llmio.config.tier import (
            LEVEL2_DEFAULT,
            LEVEL3_DEFAULT,
            TierConfig,
            TierLevelConfig,
        )

        # Level 1 default provider is "openrouter-deepseek"; override it.
        cfg = TierConfig(
            level1=TierLevelConfig(transport="claude-sdk", model="opus"),
            level2=LEVEL2_DEFAULT,
            level3=LEVEL3_DEFAULT,
        )

        create_model(level=1, tier_config=cfg)

        mock_get_provider.assert_called_once_with(provider="claude-sdk")

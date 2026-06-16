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
            create_model(transport="unknown-transport", model_level=1)
        message = str(excinfo.value)
        assert "unknown-transport" in message
        assert "claude-sdk" in message
        assert "openrouter[deepseek]" in message

    def test_invalid_model_level_raises_valueerror(self):
        for bad_level in (0, 4, -1, 99):
            with pytest.raises(ValueError) as excinfo:
                create_model(transport="claude-sdk", model_level=bad_level)
            message = str(excinfo.value)
            assert "model_level" in message
            assert str(bad_level) in message


class TestCreateModelHappyPath:
    """Valid calls delegate to ``get_provider`` with the resolved provider name."""

    @pytest.fixture
    def mock_get_provider(self, monkeypatch: pytest.MonkeyPatch) -> MagicMock:
        mock = MagicMock(return_value=MagicMock(spec=LLMProvider))
        monkeypatch.setattr("robotsix_llmio.config.factory.get_provider", mock)
        return mock

    def test_claude_sdk_level_1(self, mock_get_provider: MagicMock):
        result = create_model(transport="claude-sdk", model_level=1)
        mock_get_provider.assert_called_once_with(provider="claude-sdk")
        assert result is mock_get_provider.return_value

    def test_claude_sdk_level_2(self, mock_get_provider: MagicMock):
        result = create_model(transport="claude-sdk", model_level=2)
        mock_get_provider.assert_called_once_with(provider="claude-sdk")
        assert result is mock_get_provider.return_value

    def test_claude_sdk_level_3(self, mock_get_provider: MagicMock):
        result = create_model(transport="claude-sdk", model_level=3)
        mock_get_provider.assert_called_once_with(provider="claude-sdk")
        assert result is mock_get_provider.return_value

    def test_openrouter_deepseek_level_1(self, mock_get_provider: MagicMock):
        result = create_model(transport="openrouter[deepseek]", model_level=1)
        mock_get_provider.assert_called_once_with(provider="openrouter-deepseek")
        assert result is mock_get_provider.return_value

    def test_openrouter_deepseek_level_2(self, mock_get_provider: MagicMock):
        result = create_model(transport="openrouter[deepseek]", model_level=2)
        mock_get_provider.assert_called_once_with(provider="openrouter-deepseek")
        assert result is mock_get_provider.return_value

    def test_openrouter_deepseek_level_3(self, mock_get_provider: MagicMock):
        result = create_model(transport="openrouter[deepseek]", model_level=3)
        mock_get_provider.assert_called_once_with(provider="openrouter-deepseek")
        assert result is mock_get_provider.return_value

    def test_provider_kwargs_are_forwarded(self, mock_get_provider: MagicMock):
        create_model(
            transport="openrouter[deepseek]",
            model_level=2,
            api_key="my-key",
            base_url="https://proxy.example.com",
        )
        mock_get_provider.assert_called_once_with(
            provider="openrouter-deepseek",
            api_key="my-key",
            base_url="https://proxy.example.com",
        )

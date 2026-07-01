"""Tests for :mod:`robotsix_llmio.config.factory` — the consumer-facing
``create_model`` entry-point."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from robotsix_llmio.config.factory import create_model


class TestCreateModelValidation:
    """Input validation before delegation to ``get_provider``."""

    def test_invalid_level_raises_valueerror(self):
        for bad_level in (0, 5, -1, 99):
            with pytest.raises(ValueError) as excinfo:
                create_model(level=bad_level)
            message = str(excinfo.value)
            assert "level" in message
            assert str(bad_level) in message


class TestCreateModelHappyPath:
    """Valid calls delegate to ``get_provider`` (transport path) or
    ``get_provider_for_identifier`` (tier path)."""

    # -- Level-driven resolution (no transport) -------------------------------

    def test_level_1_no_transport_resolves_from_tier_config(
        self, mock_get_provider_for_identifier: MagicMock
    ):
        """``create_model(level=1)`` derives provider from LEVEL1_DEFAULT's
        combined identifier."""
        result = create_model(level=1)
        mock_get_provider_for_identifier.assert_called_once_with(
            "openrouter-deepseek/deepseek-v4-flash",
        )
        assert result is mock_get_provider_for_identifier.return_value

    def test_level_2_no_transport_resolves_from_tier_config(
        self, mock_get_provider_for_identifier: MagicMock
    ):
        """``create_model(level=2)`` derives provider from LEVEL2_DEFAULT's
        identifier."""
        result = create_model(level=2)
        mock_get_provider_for_identifier.assert_called_once_with(
            "openrouter-deepseek/deepseek-v4-pro",
        )
        assert result is mock_get_provider_for_identifier.return_value

    def test_level_3_no_transport_resolves_from_tier_config(
        self, mock_get_provider_for_identifier: MagicMock
    ):
        """``create_model(level=3)`` derives provider from LEVEL3_DEFAULT's
        identifier (``"claudeSDK-opus"``)."""
        result = create_model(level=3)
        mock_get_provider_for_identifier.assert_called_once_with(
            "claudeSDK-opus",
        )
        assert result is mock_get_provider_for_identifier.return_value

    # -- provider_kwargs merging ----------------------------------------------

    def test_provider_kwargs_override_tier_config_defaults(
        self, mock_get_provider_for_identifier: MagicMock
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
                model="openrouter-deepseek/deepseek-v4-flash",
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
        mock_get_provider_for_identifier.assert_called_once_with(
            "openrouter-deepseek/deepseek-v4-flash",
            base_url="https://from-tier.example.com",
            api_key="explicit-key",
        )


class TestCreateModelDefaultFallback:
    """``create_model`` falls back to the baked level defaults only when no
    user-supplied ``tier_config`` is present."""

    @pytest.mark.parametrize("level", [1, 2, 3])
    def test_no_tier_config_uses_baked_level_default(
        self, level: int, mock_get_provider_for_identifier: MagicMock
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

        mock_get_provider_for_identifier.assert_called_once_with(
            expected.model,
        )

    def test_explicit_tier_config_overrides_defaults(
        self, mock_get_provider_for_identifier: MagicMock
    ):
        """When a ``tier_config`` is supplied, its identifier is used instead of
        the baked default."""
        from robotsix_llmio.config.tier import (
            LEVEL2_DEFAULT,
            LEVEL3_DEFAULT,
            TierConfig,
            TierLevelConfig,
        )

        cfg = TierConfig(
            level1=TierLevelConfig(model="claudeSDK-opus"),
            level2=LEVEL2_DEFAULT,
            level3=LEVEL3_DEFAULT,
        )

        create_model(level=1, tier_config=cfg)

        mock_get_provider_for_identifier.assert_called_once_with(
            "claudeSDK-opus",
        )

"""Tests for :mod:`robotsix_llmio.config.transport` — alias mappings."""

from __future__ import annotations

import pytest

from robotsix_llmio.config.transport import (
    MODEL_LEVEL_TO_TIER,
    TRANSPORT_ALIASES,
    UnknownTransportError,
    provider_to_transport,
    validate_transport,
)
from robotsix_llmio.core.provider import Tier


class TestTransportAliases:
    """Transport alias → provider registry name mappings."""

    def test_claude_sdk_maps_to_claude_sdk(self):
        assert TRANSPORT_ALIASES["claude-sdk"] == "claude-sdk"

    def test_openrouter_deepseek_maps_to_openrouter_deepseek(self):
        assert TRANSPORT_ALIASES["openrouter[deepseek]"] == "openrouter-deepseek"

    def test_all_keys_are_strings(self):
        for key, value in TRANSPORT_ALIASES.items():
            assert isinstance(key, str)
            assert isinstance(value, str)


class TestModelLevelToTier:
    """model_level (int) → Tier mapping."""

    def test_level_1_maps_to_cheap(self):
        assert MODEL_LEVEL_TO_TIER[1] == Tier.CHEAP

    def test_level_2_maps_to_default(self):
        assert MODEL_LEVEL_TO_TIER[2] == Tier.DEFAULT

    def test_level_3_maps_to_default(self):
        assert MODEL_LEVEL_TO_TIER[3] == Tier.DEFAULT

    def test_all_keys_are_valid(self):
        for key, value in MODEL_LEVEL_TO_TIER.items():
            assert isinstance(key, int)
            assert key in (1, 2, 3)
            assert isinstance(value, Tier)


class TestValidateTransport:
    """``validate_transport`` accepts known aliases and rejects others."""

    def test_known_transport_passes(self):
        # Should not raise for known aliases.
        validate_transport("claude-sdk")
        validate_transport("openrouter[deepseek]")

    def test_unknown_transport_raises(self):
        with pytest.raises(UnknownTransportError, match="not-a-transport"):
            validate_transport("not-a-transport")

    def test_provider_registry_name_is_not_a_transport(self):
        """A bare provider registry name is not a valid transport alias."""
        with pytest.raises(UnknownTransportError):
            validate_transport("openrouter-deepseek")


class TestProviderToTransport:
    """``provider_to_transport`` backward-compat conversion."""

    def test_alias_passes_through_unchanged(self):
        assert provider_to_transport("openrouter[deepseek]") == "openrouter[deepseek]"

    def test_registry_name_converts_to_alias(self):
        assert provider_to_transport("openrouter-deepseek") == "openrouter[deepseek]"

    def test_claude_sdk_alias_wins_over_registry_name(self):
        """``claude-sdk`` is both an alias and a registry name; the alias wins."""
        assert provider_to_transport("claude-sdk") == "claude-sdk"

    def test_round_trip_alias_to_provider_to_alias(self):
        for alias, provider in TRANSPORT_ALIASES.items():
            assert provider_to_transport(provider) == alias

    def test_unknown_value_passes_through_unchanged(self):
        """An unrecognised value is returned unchanged so validation fails later."""
        assert provider_to_transport("mystery") == "mystery"

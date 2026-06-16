"""Tests for :mod:`robotsix_llmio.config.transport` — alias mappings."""

from __future__ import annotations

from robotsix_llmio.config.transport import MODEL_LEVEL_TO_TIER, TRANSPORT_ALIASES
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

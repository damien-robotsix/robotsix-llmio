"""End-to-end integration tests chaining the configuration *loader* to the
consumer-facing *factory*.

The loader (:func:`~robotsix_llmio.config.loader.load_tier_config`) and the
factory (:func:`~robotsix_llmio.config.factory.create_model`) are each unit
tested in isolation (``test_loader.py`` / ``test_config_factory.py``).  These
tests verify the *chain*: a config built by the loader (from a dict, or a
YAML string parsed by the caller) flows into ``create_model`` and resolves
to the expected combined identifier and merged ``provider_kwargs``.

``get_provider_for_identifier`` is monkeypatched so no real provider
import/instantiation occurs.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import yaml  # type: ignore[import-untyped]

from robotsix_llmio.config.factory import create_model
from robotsix_llmio.config.loader import load_tier_config
from robotsix_llmio.config.tier import TierConfig

# ========================================================================== #
#  loader → factory chain
# ========================================================================== #


class TestLoaderToFactoryChain:
    """``load_tier_config(...)`` → ``create_model(tier_config=...)`` end-to-end."""

    def test_custom_default_level1_resolves_to_its_identifier(
        self, mock_get_provider_for_identifier: MagicMock
    ) -> None:
        """A loaded default-slot level1 override flows into
        ``get_provider_for_identifier`` with merged kwargs."""
        cfg = load_tier_config(
            {
                "default": {
                    "level1": {
                        "model": "openrouter-deepseek/deepseek-v4-flash-latest",
                        "provider_kwargs": {"base_url": "https://proxy.example.com"},
                    }
                }
            }
        )
        assert isinstance(cfg, TierConfig)

        result = create_model(level=1, tier_config=cfg)

        mock_get_provider_for_identifier.assert_called_once_with(
            "openrouter-deepseek/deepseek-v4-flash-latest",
            base_url="https://proxy.example.com",
        )
        assert result is mock_get_provider_for_identifier.return_value

    def test_loaded_level3_resolves_to_claude_sdk(
        self, mock_get_provider_for_identifier: MagicMock
    ) -> None:
        """The baked default-slot level3 (``claudeSDK-claude-fable-5``)
        resolves with no max_tokens (Claude SDK levels carry none)."""
        cfg = load_tier_config({})

        create_model(level=3, tier_config=cfg)

        mock_get_provider_for_identifier.assert_called_once_with(
            "claudeSDK-claude-fable-5",
        )


# ========================================================================== #
#  YAML round-trip
# ========================================================================== #


class TestYamlRoundTrip:
    """YAML-string → ``yaml.safe_load`` → ``load_tier_config`` round-trip.

    NOTE: YAML parsing is the *caller's* responsibility.  ``load_tier_config``
    accepts only a plain ``dict``; it has no YAML loader.  These tests
    demonstrate the supported integration pattern — parse YAML to a dict
    externally, then feed that dict to the loader.
    """

    def test_yaml_string_loads_into_tier_config(self) -> None:
        """A YAML string parsed to a dict and fed to ``load_tier_config``
        produces a ``TierConfig`` whose fields match the YAML."""
        yaml_text = """
        default:
          level2:
            model: "claudeSDK-sonnet"
        fallback:
          level3:
            model: "openrouter-deepseek/deepseek-v4-pro"
        failover:
          window_seconds: 300
        """
        cfg = load_tier_config(yaml.safe_load(yaml_text))

        assert cfg.default.level2.model == "claudeSDK-sonnet"
        assert cfg.fallback.level3.model == "openrouter-deepseek/deepseek-v4-pro"
        assert cfg.failover.window_seconds == 300.0

    def test_yaml_with_provider_kwargs_round_trips(self) -> None:
        """``provider_kwargs`` parsed from YAML survive into the ``TierConfig``."""
        yaml_text = """
        default:
          level1:
            model: "claudeSDK-opus"
            provider_kwargs:
              timeout: 30
              base_url: "https://proxy.example.com"
        """
        cfg = load_tier_config(yaml.safe_load(yaml_text))

        assert cfg.default.level1.model == "claudeSDK-opus"
        assert cfg.default.level1.provider_kwargs == {
            "timeout": 30,
            "base_url": "https://proxy.example.com",
        }

    def test_yaml_round_trip_chains_to_factory(
        self, mock_get_provider_for_identifier: MagicMock
    ) -> None:
        """Full chain: YAML string → dict → loader → ``create_model``."""
        yaml_text = """
        fallback:
          level1:
            model: "openrouter-deepseek/deepseek-v4-flash-latest"
        """
        cfg = load_tier_config(yaml.safe_load(yaml_text))

        create_model(level=1, tier_config=cfg)

        # The default slot is untouched by a fallback-slot override, and no
        # failover is armed — level 1 still resolves claudeSDK-haiku.
        mock_get_provider_for_identifier.assert_called_once_with("claudeSDK-haiku")
        # The override IS present in the fallback slot, inheriting the baked
        # max_tokens and preferred_provider it did not restate.
        assert (
            cfg.fallback.level1.model == "openrouter-deepseek/deepseek-v4-flash-latest"
        )
        assert cfg.fallback.level1.max_tokens == 16384
        assert cfg.fallback.level1.provider_kwargs == {
            "preferred_provider": "DeepInfra"
        }

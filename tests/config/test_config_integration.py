"""End-to-end integration tests chaining the configuration *loader* to the
consumer-facing *factory*.

The loader (:func:`~robotsix_llmio.config.loader.load_tier_config`) and the
factory (:func:`~robotsix_llmio.config.factory.create_model`) are each unit
tested in isolation (``test_loader.py`` / ``test_config_factory.py``).  These
tests verify the *chain*: a config built by the loader (from a dict, env vars,
or a YAML string parsed by the caller) flows into ``create_model`` and resolves
to the expected combined identifier and merged ``provider_kwargs``.

``get_provider_for_identifier`` is monkeypatched so no real provider
import/instantiation occurs.
"""

from __future__ import annotations

import os
from unittest.mock import MagicMock

import pytest
import yaml  # type: ignore[import-untyped]

from robotsix_llmio.config.factory import create_model
from robotsix_llmio.config.loader import load_tier_config
from robotsix_llmio.config.tier import TierConfig
from robotsix_llmio.core.provider import LLMProvider


@pytest.fixture(autouse=True)
def clean_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """Remove all LLMIO_* variables so tests start from a known-clean env."""
    for key in tuple(os.environ):
        if key.startswith("LLMIO_"):
            monkeypatch.delenv(key, raising=False)


@pytest.fixture
def mock_get_provider_for_identifier(
    monkeypatch: pytest.MonkeyPatch,
) -> MagicMock:
    """Patch ``get_provider_for_identifier`` as referenced by the factory module."""
    mock = MagicMock(return_value=MagicMock(spec=LLMProvider))
    monkeypatch.setattr(
        "robotsix_llmio.config.factory.get_provider_for_identifier", mock
    )
    return mock


# ========================================================================== #
#  loader → factory chain
# ========================================================================== #


class TestLoaderToFactoryChain:
    """``load_tier_config(...)`` → ``create_model(tier_config=...)`` end-to-end."""

    def test_custom_level1_config_resolves_to_its_identifier(
        self, mock_get_provider_for_identifier: MagicMock
    ) -> None:
        """A loaded level1 config with a combined ``model`` identifier flows
        into ``get_provider_for_identifier`` with merged kwargs."""
        cfg = load_tier_config(
            {
                "level1": {
                    "model": "openrouter[deepseek]-deepseek/deepseek-v4-flash",
                    "provider_kwargs": {"base_url": "https://proxy.example.com"},
                }
            }
        )
        assert isinstance(cfg, TierConfig)

        result = create_model(level=1, tier_config=cfg)

        mock_get_provider_for_identifier.assert_called_once_with(
            "openrouter[deepseek]-deepseek/deepseek-v4-flash",
            base_url="https://proxy.example.com",
        )
        assert result is mock_get_provider_for_identifier.return_value

    def test_level3_loaded_config_resolves_to_claude_sdk(
        self, mock_get_provider_for_identifier: MagicMock
    ) -> None:
        """A loaded level3 ``claudeSDK-opus`` config resolves correctly."""
        cfg = load_tier_config(
            {
                "level1": {
                    "model": "openrouter[deepseek]-deepseek/deepseek-v4-flash",
                },
                "level3": {"model": "claudeSDK-opus"},
            }
        )

        create_model(level=3, tier_config=cfg)

        mock_get_provider_for_identifier.assert_called_once_with(
            "claudeSDK-opus",
        )


# ========================================================================== #
#  YAML round-trip
# ========================================================================== #


class TestYamlRoundTrip:
    """YAML-string → ``yaml.safe_load`` → ``load_tier_config`` round-trip.

    NOTE: YAML parsing is the *caller's* responsibility.  ``load_tier_config``
    accepts only a plain ``dict`` (plus env vars); it has no YAML loader.  These
    tests demonstrate the supported integration pattern — parse YAML to a dict
    externally, then feed that dict to the loader.
    """

    def test_yaml_string_loads_into_tier_config(self) -> None:
        """A YAML string parsed to a dict and fed to ``load_tier_config``
        produces a ``TierConfig`` whose fields match the YAML."""
        yaml_text = """
        level1:
          model: "openrouter[deepseek]-deepseek/deepseek-v4-flash"
        level2:
          model: "openrouter[deepseek]-deepseek/deepseek-v4-pro"
        level3:
          model: "claudeSDK-opus"
        """
        config_dict = yaml.safe_load(yaml_text)

        cfg = load_tier_config(config_dict)

        assert cfg.level1.model == "openrouter[deepseek]-deepseek/deepseek-v4-flash"
        assert cfg.level2.model == "openrouter[deepseek]-deepseek/deepseek-v4-pro"
        assert cfg.level3.model == "claudeSDK-opus"

    def test_yaml_with_provider_kwargs_round_trips(self) -> None:
        """``provider_kwargs`` parsed from YAML survive into the ``TierConfig``."""
        yaml_text = """
        level1:
          model: "claudeSDK-opus"
          provider_kwargs:
            timeout: 30
            base_url: "https://proxy.example.com"
        """
        config_dict = yaml.safe_load(yaml_text)

        cfg = load_tier_config(config_dict)

        assert cfg.level1.model == "claudeSDK-opus"
        assert cfg.level1.provider_kwargs == {
            "timeout": 30,
            "base_url": "https://proxy.example.com",
        }

    def test_yaml_round_trip_chains_to_factory(
        self, mock_get_provider_for_identifier: MagicMock
    ) -> None:
        """Full chain: YAML string → dict → loader → ``create_model``."""
        yaml_text = """
        level1:
          model: "openrouter[deepseek]-deepseek/deepseek-v4-flash"
        """
        cfg = load_tier_config(yaml.safe_load(yaml_text))

        create_model(level=1, tier_config=cfg)

        mock_get_provider_for_identifier.assert_called_once_with(
            "openrouter[deepseek]-deepseek/deepseek-v4-flash",
        )

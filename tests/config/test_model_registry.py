"""Tests for the per-provider named-model registry and validation.

Covers:
- ``PROVIDER_MODELS`` keys and value types
- ``validate_model`` for known provider + known model
- ``validate_model`` raises ``UnknownModelError`` for known provider + unknown model
- ``validate_model`` silently passes for unknown provider
- ``UnknownModelError`` inheritance and message format
- ``TierLevelConfig`` integration — model validation at construction time
"""

from __future__ import annotations

import pytest

from robotsix_llmio.config.model_registry import (
    PROVIDER_MODELS,
    UnknownModelError,
    validate_model,
)
from robotsix_llmio.config.tier import (
    LEVEL1_DEFAULT,
    LEVEL2_DEFAULT,
    LEVEL3_DEFAULT,
    TierConfig,
    TierLevelConfig,
)
from robotsix_llmio.exceptions import RobotsixLLMIOError

# ========================================================================== #
#  PROVIDER_MODELS registry                                                   #
# ========================================================================== #


def test_registry_has_expected_keys():
    """Registry keys cover all known providers."""
    assert set(PROVIDER_MODELS.keys()) == {"openrouter-deepseek", "claude-sdk"}


def test_registry_values_are_frozensets():
    """Every value is a :class:`frozenset` of model name strings."""
    for models in PROVIDER_MODELS.values():
        assert isinstance(models, frozenset)
        for m in models:
            assert isinstance(m, str)


def test_openrouter_deepseek_models():
    """The openrouter-deepseek provider has the expected model set."""
    models = PROVIDER_MODELS["openrouter-deepseek"]
    assert models == frozenset(
        {"deepseek/deepseek-v4-pro", "deepseek/deepseek-v4-flash"}
    )


def test_claude_sdk_models():
    """The claude-sdk provider has the expected model set."""
    models = PROVIDER_MODELS["claude-sdk"]
    assert models == frozenset({"opus", "haiku", "sonnet"})


# ========================================================================== #
#  validate_model                                                             #
# ========================================================================== #


# -- known provider + known model -------------------------------------------


@pytest.mark.parametrize(
    "provider, model",
    [
        ("openrouter-deepseek", "deepseek/deepseek-v4-pro"),
        ("openrouter-deepseek", "deepseek/deepseek-v4-flash"),
        ("claude-sdk", "opus"),
        ("claude-sdk", "haiku"),
        ("claude-sdk", "sonnet"),
    ],
)
def test_validate_model_known_provider_known_model(provider, model):
    """``validate_model`` returns ``None`` for known provider + known model."""
    validate_model(provider, model)  # does not raise


# -- known provider + unknown model -----------------------------------------


def test_validate_model_unknown_model_openrouter():
    """Raises ``UnknownModelError`` for openrouter-deepseek with bogus model."""
    with pytest.raises(UnknownModelError) as exc_info:
        validate_model("openrouter-deepseek", "gpt-4")
    msg = str(exc_info.value)
    assert "openrouter-deepseek" in msg
    assert "gpt-4" in msg
    assert "Known models:" in msg
    assert "deepseek/deepseek-v4-flash" in msg
    assert "deepseek/deepseek-v4-pro" in msg


def test_validate_model_unknown_model_claude():
    """Raises ``UnknownModelError`` for claude-sdk with bogus model."""
    with pytest.raises(UnknownModelError) as exc_info:
        validate_model("claude-sdk", "gpt-4")
    msg = str(exc_info.value)
    assert "claude-sdk" in msg
    assert "gpt-4" in msg
    assert "Known models:" in msg
    assert "haiku" in msg
    assert "opus" in msg
    assert "sonnet" in msg


# -- unknown provider -------------------------------------------------------


def test_validate_model_unknown_provider_does_not_raise():
    """Unknown providers skip model validation — returns ``None``."""
    validate_model("unknown-provider", "anything")  # does not raise


def test_validate_model_unknown_provider_empty_model():
    """Unknown providers with empty model string also pass."""
    validate_model("unknown-x", "")  # does not raise


# ========================================================================== #
#  UnknownModelError                                                          #
# ========================================================================== #


def test_unknown_model_error_is_robotsix_error():
    """``UnknownModelError`` inherits from ``RobotsixLLMIOError``."""
    assert issubclass(UnknownModelError, RobotsixLLMIOError)


def test_unknown_model_error_message_contains_provider():
    """Error message names the provider."""
    err = UnknownModelError("test message with provider-x")
    assert "provider-x" in str(err)


def test_unknown_model_error_message_contains_model():
    """Error message names the unknown model."""
    err = UnknownModelError("test message with model-y")
    assert "model-y" in str(err)


def test_unknown_model_error_is_exception():
    """``UnknownModelError`` is a standard exception."""
    assert issubclass(UnknownModelError, Exception)


# ========================================================================== #
#  TierLevelConfig integration                                                #
# ========================================================================== #


def test_tier_level_config_accepts_any_model_with_known_prefix():
    """Constructing with a known provider prefix and any model name
    succeeds (model-name validation is the backend's concern)."""
    cfg = TierLevelConfig(model="openrouter[deepseek]-bogus-model")
    assert cfg.model == "openrouter[deepseek]-bogus-model"
    assert cfg.provider == "openrouter"
    assert cfg.model_name == "bogus-model"


def test_tier_level_config_accepts_any_model_claude():
    """Any model name with the claudeSDK prefix succeeds."""
    cfg = TierLevelConfig(model="claudeSDK-nonexistent")
    assert cfg.model == "claudeSDK-nonexistent"
    assert cfg.provider == "claudeSDK"
    assert cfg.model_name == "nonexistent"


def test_tier_level_config_accepts_known_model_openrouter():
    """Constructing with a combined identifier for openrouter[deepseek]
    succeeds."""
    cfg = TierLevelConfig(model="openrouter[deepseek]-deepseek/deepseek-v4-flash")
    assert cfg.model == "openrouter[deepseek]-deepseek/deepseek-v4-flash"
    assert cfg.model_name == "deepseek/deepseek-v4-flash"


def test_tier_level_config_accepts_known_model_claude():
    """Constructing with a combined identifier for claudeSDK succeeds."""
    cfg = TierLevelConfig(model="claudeSDK-sonnet")
    assert cfg.model == "claudeSDK-sonnet"
    assert cfg.model_name == "sonnet"


def test_validate_model_skips_unknown_provider():
    """``validate_model`` silently passes for an unknown provider registry name."""
    # Should not raise — unknown provider is validated elsewhere.
    validate_model("unknown-x", "anything-goes")


def test_baked_defaults_construct_without_error():
    """The three module-level baked defaults validate successfully."""
    assert LEVEL1_DEFAULT.model == "openrouter[deepseek]-deepseek/deepseek-v4-flash"
    assert LEVEL1_DEFAULT.provider == "openrouter"
    assert LEVEL1_DEFAULT.model_name == "deepseek/deepseek-v4-flash"
    assert LEVEL2_DEFAULT.model == "openrouter[deepseek]-deepseek/deepseek-v4-pro"
    assert LEVEL2_DEFAULT.provider == "openrouter"
    assert LEVEL2_DEFAULT.model_name == "deepseek/deepseek-v4-pro"
    assert LEVEL3_DEFAULT.model == "claudeSDK-opus"
    assert LEVEL3_DEFAULT.provider == "claudeSDK"
    assert LEVEL3_DEFAULT.model_name == "opus"


def test_tier_config_model_validate_accepts_any_model():
    """``TierConfig.model_validate`` with a known prefix accepts any model
    name (model-name validation is the backend's concern)."""
    data = {
        "level1": {"model": "openrouter[deepseek]-bogus"},
    }
    cfg = TierConfig.model_validate(data)
    assert cfg.level1.model == "openrouter[deepseek]-bogus"
    assert cfg.level1.model_name == "bogus"


def test_tier_config_model_validate_accepts_valid_model():
    """``TierConfig.model_validate`` with valid combined identifier succeeds."""
    data = {
        "level1": {
            "model": "openrouter[deepseek]-deepseek/deepseek-v4-flash",
        },
    }
    cfg = TierConfig.model_validate(data)
    assert cfg.level1.model == "openrouter[deepseek]-deepseek/deepseek-v4-flash"
    assert cfg.level1.model_name == "deepseek/deepseek-v4-flash"


# ========================================================================== #
#  Re-exports                                                                 #
# ========================================================================== #


def test_config_package_reexports_provider_models():
    """``PROVIDER_MODELS`` is importable from ``robotsix_llmio.config``."""
    from robotsix_llmio.config import PROVIDER_MODELS as PM

    assert PM is PROVIDER_MODELS


def test_config_package_reexports_unknown_model_error():
    """``UnknownModelError`` is importable from ``robotsix_llmio.config``."""
    from robotsix_llmio.config import UnknownModelError as UME

    assert UME is UnknownModelError


def test_config_package_reexports_validate_model():
    """``validate_model`` is importable from ``robotsix_llmio.config``."""
    from robotsix_llmio.config import validate_model as vm

    assert vm is validate_model


def test_core_package_reexports_provider_models():
    """``PROVIDER_MODELS`` is importable from ``robotsix_llmio.core``."""
    from robotsix_llmio.core import PROVIDER_MODELS as PM

    assert PM is PROVIDER_MODELS


def test_core_package_reexports_unknown_model_error():
    """``UnknownModelError`` is importable from ``robotsix_llmio.core``."""
    from robotsix_llmio.core import UnknownModelError as UME

    assert UME is UnknownModelError


def test_core_package_reexports_validate_model():
    """``validate_model`` is importable from ``robotsix_llmio.core``."""
    from robotsix_llmio.core import validate_model as vm

    assert vm is validate_model

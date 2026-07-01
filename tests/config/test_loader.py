"""Tests for the tier configuration loader.

Covers:
- ``load_tier_config()`` with no arguments / no env vars
- Environment-variable overrides (``LLMIO_LEVEL{n}_MODEL``)
- ``*_PROVIDER_KWARGS`` parsing (valid and invalid JSON)
- Explicit ``config_dict`` overriding env vars
- Partial ``config_dict`` merge
- Omitted tiers (including ``level1``) → baked defaults
- Re-export smoke tests from ``robotsix_llmio.config`` and
  ``robotsix_llmio.core``
"""

from __future__ import annotations

import pytest

from robotsix_llmio.config.loader import TierConfigLoadError, load_tier_config
from robotsix_llmio.config.tier import (
    LEVEL1_DEFAULT,
    LEVEL2_DEFAULT,
    LEVEL3_DEFAULT,
    LEVEL4_DEFAULT,
    TierLevelConfig,
)

# ========================================================================== #
#  Helpers
# ========================================================================== #


def set_env(monkeypatch: pytest.MonkeyPatch, **kwargs: str) -> None:
    """Convenience: set multiple env vars at once."""
    for k, v in kwargs.items():
        monkeypatch.setenv(k, v)


# ========================================================================== #
#  No arguments / no env
# ========================================================================== #


def test_no_args_returns_baked_defaults():
    """With no env vars and no explicit dict, all four tiers fall back to
    their baked defaults."""
    cfg = load_tier_config()
    assert cfg.level1 == LEVEL1_DEFAULT
    assert cfg.level2 == LEVEL2_DEFAULT
    assert cfg.level3 == LEVEL3_DEFAULT
    assert cfg.level4 == LEVEL4_DEFAULT


# ========================================================================== #
#  Explicit dict
# ========================================================================== #


def test_explicit_dict_level1_only():
    """Supplying only level1 gives baked defaults for level2/level3."""
    cfg = load_tier_config({"level1": {"model": "claudeSDK-opus"}})
    assert cfg.level1.model == "claudeSDK-opus"
    assert cfg.level1.provider == "claudeSDK"
    assert cfg.level1.model_name == "opus"
    assert cfg.level2 == LEVEL2_DEFAULT
    assert cfg.level3 == LEVEL3_DEFAULT


def test_explicit_dict_full():
    """All three tiers can be supplied via explicit dict."""
    cfg = load_tier_config(
        {
            "level1": {"model": "claudeSDK-haiku"},
            "level2": {
                "model": "openrouter-deepseek/deepseek-v4-pro",
            },
            "level3": {"model": "claudeSDK-opus"},
        }
    )
    assert cfg.level1.model_name == "haiku"
    assert cfg.level2.model == "openrouter-deepseek/deepseek-v4-pro"
    assert cfg.level3.model_name == "opus"


def test_explicit_dict_none_passed():
    """Passing ``config_dict=None`` is the same as omitting it — baked
    defaults for every tier."""
    cfg = load_tier_config(None)
    assert cfg.level1 == LEVEL1_DEFAULT
    assert cfg.level2 == LEVEL2_DEFAULT
    assert cfg.level3 == LEVEL3_DEFAULT


# ========================================================================== #
#  Environment variable overrides
# ========================================================================== #


def test_env_level1_model(monkeypatch: pytest.MonkeyPatch):
    """``LLMIO_LEVEL1_MODEL`` provides the combined identifier verbatim."""
    set_env(
        monkeypatch,
        LLMIO_LEVEL1_MODEL="claudeSDK-haiku",
    )
    cfg = load_tier_config()
    assert cfg.level1.model == "claudeSDK-haiku"
    assert cfg.level1.provider == "claudeSDK"
    assert cfg.level1.model_name == "haiku"


def test_env_level2_model_override(monkeypatch: pytest.MonkeyPatch):
    """``LLMIO_LEVEL2_MODEL`` overrides the baked level2 model."""
    set_env(
        monkeypatch,
        LLMIO_LEVEL1_MODEL="claudeSDK-opus",
        LLMIO_LEVEL2_MODEL="claudeSDK-sonnet",
    )
    cfg = load_tier_config()
    assert cfg.level2.model == "claudeSDK-sonnet"
    assert cfg.level2.provider == "claudeSDK"
    assert cfg.level2.model_name == "sonnet"


def test_env_level3_model_override(monkeypatch: pytest.MonkeyPatch):
    """``LLMIO_LEVEL3_MODEL`` overrides the baked level3 model."""
    set_env(
        monkeypatch,
        LLMIO_LEVEL1_MODEL="claudeSDK-opus",
        LLMIO_LEVEL3_MODEL="claudeSDK-haiku",
    )
    cfg = load_tier_config()
    assert cfg.level3.model == "claudeSDK-haiku"
    assert cfg.level3.model_name == "haiku"


def test_env_level4_model_override(monkeypatch: pytest.MonkeyPatch):
    """``LLMIO_LEVEL4_MODEL`` overrides the baked level4 model."""
    set_env(
        monkeypatch,
        LLMIO_LEVEL4_MODEL="claudeSDK-opus",
    )
    cfg = load_tier_config()
    assert cfg.level4.model == "claudeSDK-opus"
    assert cfg.level4.model_name == "opus"
    # Untouched tiers keep their baked defaults.
    assert cfg.level1 == LEVEL1_DEFAULT
    assert cfg.level3 == LEVEL3_DEFAULT


def test_env_model_must_be_valid_identifier(monkeypatch: pytest.MonkeyPatch):
    """An unknown provider prefix in the env model raises TierConfigLoadError."""
    set_env(
        monkeypatch,
        LLMIO_LEVEL1_MODEL="notAPrefix-opus",
    )
    with pytest.raises(TierConfigLoadError):
        load_tier_config()


# ========================================================================== #
#  PROVIDER_KWARGS — valid JSON
# ========================================================================== #


def test_env_provider_kwargs_valid_json(monkeypatch: pytest.MonkeyPatch):
    """``*_PROVIDER_KWARGS`` with valid JSON is parsed into a dict."""
    set_env(
        monkeypatch,
        LLMIO_LEVEL1_MODEL="claudeSDK-opus",
        LLMIO_LEVEL2_PROVIDER_KWARGS='{"timeout": 30, "base_url": "https://x"}',
    )
    cfg = load_tier_config()
    assert cfg.level2.provider_kwargs == {"timeout": 30, "base_url": "https://x"}


def test_env_provider_kwargs_empty_object(monkeypatch: pytest.MonkeyPatch):
    """``*_PROVIDER_KWARGS`` can be an empty JSON object."""
    set_env(
        monkeypatch,
        LLMIO_LEVEL1_MODEL="claudeSDK-opus",
        LLMIO_LEVEL3_PROVIDER_KWARGS="{}",
    )
    cfg = load_tier_config()
    assert cfg.level3.provider_kwargs == {}


def test_env_provider_kwargs_merge_with_baked_model(
    monkeypatch: pytest.MonkeyPatch,
):
    """Setting only ``provider_kwargs`` via env keeps the baked model."""
    set_env(
        monkeypatch,
        LLMIO_LEVEL1_MODEL="claudeSDK-opus",
        LLMIO_LEVEL2_PROVIDER_KWARGS='{"timeout": 30}',
    )
    cfg = load_tier_config()
    # Baked level2 model is preserved; provider_kwargs comes from env.
    assert cfg.level2.model == LEVEL2_DEFAULT.model
    assert cfg.level2.provider_kwargs == {"timeout": 30}


# ========================================================================== #
#  PROVIDER_KWARGS — invalid JSON
# ========================================================================== #


def test_env_provider_kwargs_invalid_json(monkeypatch: pytest.MonkeyPatch):
    """Invalid JSON in a ``*_PROVIDER_KWARGS`` var raises TierConfigLoadError."""
    set_env(
        monkeypatch,
        LLMIO_LEVEL1_MODEL="claudeSDK-opus",
        LLMIO_LEVEL1_PROVIDER_KWARGS="not json",
    )
    with pytest.raises(TierConfigLoadError) as exc_info:
        load_tier_config()
    msg = str(exc_info.value)
    assert "LLMIO_LEVEL1_PROVIDER_KWARGS" in msg


def test_env_provider_kwargs_json_array(monkeypatch: pytest.MonkeyPatch):
    """A JSON array is not a valid value for provider_kwargs."""
    set_env(
        monkeypatch,
        LLMIO_LEVEL1_MODEL="claudeSDK-opus",
        LLMIO_LEVEL3_PROVIDER_KWARGS="[1, 2, 3]",
    )
    with pytest.raises(TierConfigLoadError) as exc_info:
        load_tier_config()
    msg = str(exc_info.value)
    assert "LLMIO_LEVEL3_PROVIDER_KWARGS" in msg
    assert "JSON object" in msg or "list" in msg


# ========================================================================== #
#  Explicit dict overrides env vars
# ========================================================================== #


def test_explicit_dict_overrides_env(monkeypatch: pytest.MonkeyPatch):
    """config_dict has higher precedence than env vars."""
    set_env(
        monkeypatch,
        LLMIO_LEVEL1_MODEL="claudeSDK-opus",
    )
    cfg = load_tier_config(
        {
            "level1": {
                "model": "openrouter-deepseek/deepseek-v4-pro",
            }
        }
    )
    assert cfg.level1.model == "openrouter-deepseek/deepseek-v4-pro"
    assert cfg.level1.provider == "openrouter"
    assert cfg.level1.model_name == "deepseek/deepseek-v4-pro"


def test_explicit_dict_partial_overrides_env(monkeypatch: pytest.MonkeyPatch):
    """A partial config_dict overrides only the specified fields."""
    set_env(
        monkeypatch,
        LLMIO_LEVEL1_MODEL="claudeSDK-opus",
        LLMIO_LEVEL1_PROVIDER_KWARGS='{"timeout": 10}',
    )
    # Override only the model field with a different combined identifier.
    cfg = load_tier_config({"level1": {"model": "claudeSDK-haiku"}})
    assert cfg.level1.model == "claudeSDK-haiku"
    assert cfg.level1.model_name == "haiku"
    # provider_kwargs from env still apply.
    assert cfg.level1.provider_kwargs == {"timeout": 10}


# ========================================================================== #
#  Partial config_dict — merge correctness
# ========================================================================== #


def test_partial_dict_only_level1():
    """Only level1 in dict → level2/3 come from baked defaults."""
    cfg = load_tier_config({"level1": {"model": "claudeSDK-opus"}})
    assert cfg.level2 == LEVEL2_DEFAULT
    assert cfg.level3 == LEVEL3_DEFAULT


def test_partial_dict_only_level2_model(monkeypatch: pytest.MonkeyPatch):
    """Setting only level2.model in dict overrides the baked model."""
    set_env(
        monkeypatch,
        LLMIO_LEVEL1_MODEL="claudeSDK-opus",
    )
    cfg = load_tier_config(
        {"level2": {"model": "openrouter-deepseek/deepseek-v4-flash"}}
    )
    assert cfg.level2.model == "openrouter-deepseek/deepseek-v4-flash"
    assert cfg.level2.model_name == "deepseek/deepseek-v4-flash"


# ========================================================================== #
#  Omitted level1 falls back to its baked default
# ========================================================================== #


def test_missing_level1_uses_default():
    """If level1 is missing from both env and dict, it falls back to
    ``LEVEL1_DEFAULT`` (no error)."""
    cfg = load_tier_config({})
    assert cfg.level1 == LEVEL1_DEFAULT


def test_missing_level1_partial_other_tiers(monkeypatch: pytest.MonkeyPatch):
    """Providing only level2/level3 leaves level1 at its baked default."""
    set_env(
        monkeypatch,
        LLMIO_LEVEL2_MODEL="claudeSDK-haiku",
    )
    cfg = load_tier_config({"level3": {"model": "claudeSDK-sonnet"}})
    assert cfg.level1 == LEVEL1_DEFAULT
    assert cfg.level2.model == "claudeSDK-haiku"
    assert cfg.level3.model == "claudeSDK-sonnet"


# ========================================================================== #
#  Re-export smoke tests
# ========================================================================== #


def test_reexport_from_config_package():
    """``load_tier_config`` is importable from ``robotsix_llmio.config``."""
    from robotsix_llmio.config import TierConfigLoadError as T1
    from robotsix_llmio.config import load_tier_config as f1

    assert f1 is load_tier_config
    assert T1 is TierConfigLoadError


def test_reexport_from_core_package():
    """``load_tier_config`` is importable from ``robotsix_llmio.core``."""
    from robotsix_llmio.core import TierConfigLoadError as T1
    from robotsix_llmio.core import load_tier_config as f1

    assert f1 is load_tier_config
    assert T1 is TierConfigLoadError


# ========================================================================== #
#  Non-dict tier values in config_dict  (_to_dict path)
# ========================================================================== #


def test_config_dict_with_tier_level_config_object(monkeypatch: pytest.MonkeyPatch):
    """Passing a TierLevelConfig object for a tier exercises _to_dict()."""
    set_env(
        monkeypatch,
        LLMIO_LEVEL1_MODEL="claudeSDK-opus",
    )
    # Pass level2 as a TierLevelConfig object instead of a dict.
    cfg = load_tier_config(
        {
            "level1": {"model": "claudeSDK-haiku"},
            "level2": TierLevelConfig(model="claudeSDK-sonnet"),
        }
    )
    assert cfg.level1.model_name == "haiku"
    assert cfg.level2.model == "claudeSDK-sonnet"
    assert cfg.level2.model_name == "sonnet"


def test_to_dict_fallback_to_pydantic_v1_dict(monkeypatch: pytest.MonkeyPatch):
    """_to_dict falls back to .dict() when .model_dump() is absent."""

    class V1Style:
        def dict(self) -> dict[str, str]:
            return {"model": "claudeSDK-sonnet"}

    set_env(
        monkeypatch,
        LLMIO_LEVEL1_MODEL="claudeSDK-opus",
    )
    cfg = load_tier_config(
        {
            "level1": {"model": "claudeSDK-haiku"},
            "level2": V1Style(),
        }
    )
    assert cfg.level2.model == "claudeSDK-sonnet"
    assert cfg.level2.model_name == "sonnet"


def test_to_dict_unmergeable_value_raises(monkeypatch: pytest.MonkeyPatch):
    """A value neither a dict nor a pydantic model raises TierConfigLoadError."""
    set_env(
        monkeypatch,
        LLMIO_LEVEL1_MODEL="claudeSDK-opus",
    )
    with pytest.raises(TierConfigLoadError) as exc_info:
        load_tier_config(
            {
                "level1": {"model": "claudeSDK-haiku"},
                "level2": 42,
            }
        )  # int — not mergeable
    msg = str(exc_info.value)
    assert "Cannot merge" in msg or "int" in msg

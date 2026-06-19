"""Tests for the tier configuration loader.

Covers:
- ``load_tier_config()`` with no arguments / no env vars
- Environment-variable overrides (new-style ``_TRANSPORT`` / ``_MODEL``)
- ``_TRANSPORT`` taking precedence over the backward-compat ``_PROVIDER``
- ``*_PROVIDER_KWARGS`` parsing (valid and invalid JSON)
- Explicit ``config_dict`` overriding env vars, including old-shape dicts
- Partial ``config_dict`` merge
- Missing ``level1`` → ``TierConfigLoadError``
- Unknown transport → ``TierConfigLoadError`` (transport is combined with
  model; an unknown prefix surfaces as a validation error)
- Legacy environment variables → ``FutureWarning`` + correct mapping
- Legacy variables silently ignored when new-style is set
- ``LLMIO_PROVIDER`` fallback to all levels lacking an explicit transport
- Re-export smoke tests from ``robotsix_llmio.config`` and
  ``robotsix_llmio.core``
"""

from __future__ import annotations

import os
import warnings

import pytest

from robotsix_llmio.config.loader import TierConfigLoadError, load_tier_config
from robotsix_llmio.config.tier import (
    LEVEL2_DEFAULT,
    LEVEL3_DEFAULT,
    TierLevelConfig,
)

# ========================================================================== #
#  Helpers
# ========================================================================== #


@pytest.fixture(autouse=True)
def clean_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """Remove all LLMIO_* variables so tests start from a known-clean env."""
    for key in tuple(os.environ):
        if key.startswith("LLMIO_"):
            monkeypatch.delenv(key, raising=False)


def set_env(monkeypatch: pytest.MonkeyPatch, **kwargs: str) -> None:
    """Convenience: set multiple env vars at once."""
    for k, v in kwargs.items():
        monkeypatch.setenv(k, v)


# ========================================================================== #
#  No arguments / no env
# ========================================================================== #


def test_no_args_raises():
    """With no env vars and no explicit dict, level1 is missing → error."""
    with pytest.raises(TierConfigLoadError) as exc_info:
        load_tier_config()
    msg = str(exc_info.value)
    assert "level1" in msg.lower()
    # The original pydantic error should be chained.
    assert exc_info.value.__cause__ is not None


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
                "model": "openrouter[deepseek]-deepseek/deepseek-v4-pro",
            },
            "level3": {"model": "claudeSDK-opus"},
        }
    )
    assert cfg.level1.model_name == "haiku"
    assert cfg.level2.model == "openrouter[deepseek]-deepseek/deepseek-v4-pro"
    assert cfg.level3.model_name == "opus"


def test_explicit_dict_old_transport_model_shape():
    """An explicit dict with the old ``transport`` + ``model`` keys still
    loads via ``_normalise_old_shape``."""
    cfg = load_tier_config(
        {
            "level1": {
                "transport": "openrouter[deepseek]",
                "model": "deepseek/deepseek-v4-flash",
            }
        }
    )
    assert cfg.level1.model == "openrouter[deepseek]-deepseek/deepseek-v4-flash"
    assert cfg.level1.provider == "openrouter"
    assert cfg.level1.model_name == "deepseek/deepseek-v4-flash"


def test_explicit_dict_old_provider_key():
    """An explicit dict with the old ``provider`` key still loads."""
    cfg = load_tier_config(
        {
            "level1": {
                "provider": "openrouter-deepseek",
                "model": "deepseek/deepseek-v4-flash",
            }
        }
    )
    assert cfg.level1.model == "openrouter[deepseek]-deepseek/deepseek-v4-flash"
    assert cfg.level1.provider == "openrouter"
    assert cfg.level1.model_name == "deepseek/deepseek-v4-flash"


def test_explicit_dict_none_passed():
    """Passing ``config_dict=None`` is the same as omitting it."""
    # Still raises because level1 is missing.
    with pytest.raises(TierConfigLoadError):
        load_tier_config(None)


# ========================================================================== #
#  Environment variable overrides (new-style)
# ========================================================================== #


def test_env_level1_transport_and_model(monkeypatch: pytest.MonkeyPatch):
    """``LLMIO_LEVEL1_TRANSPORT`` + ``_MODEL`` are combined into the
    ``model`` identifier."""
    set_env(
        monkeypatch,
        LLMIO_LEVEL1_TRANSPORT="claude-sdk",
        LLMIO_LEVEL1_MODEL="haiku",
    )
    cfg = load_tier_config()
    assert cfg.level1.model == "claudeSDK-haiku"
    assert cfg.level1.provider == "claudeSDK"
    assert cfg.level1.model_name == "haiku"


def test_env_level2_transport_override(monkeypatch: pytest.MonkeyPatch):
    """``LLMIO_LEVEL2_TRANSPORT`` / ``_MODEL`` override the baked level2."""
    set_env(
        monkeypatch,
        LLMIO_LEVEL1_TRANSPORT="claude-sdk",
        LLMIO_LEVEL1_MODEL="opus",
        LLMIO_LEVEL2_TRANSPORT="claude-sdk",
        LLMIO_LEVEL2_MODEL="sonnet",
    )
    cfg = load_tier_config()
    assert cfg.level2.model == "claudeSDK-sonnet"
    assert cfg.level2.provider == "claudeSDK"
    assert cfg.level2.model_name == "sonnet"


def test_env_level2_model_override_keeps_baked_transport(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Overriding only level2.model (as bare model name) combines with the
    baked level2 transport prefix."""
    set_env(
        monkeypatch,
        LLMIO_LEVEL1_TRANSPORT="claude-sdk",
        LLMIO_LEVEL1_MODEL="opus",
        LLMIO_LEVEL2_MODEL="deepseek/deepseek-v4-flash",
    )
    cfg = load_tier_config()
    assert cfg.level2.model == "openrouter[deepseek]-deepseek/deepseek-v4-flash"
    assert cfg.level2.model_name == "deepseek/deepseek-v4-flash"


def test_env_level3_model_override(monkeypatch: pytest.MonkeyPatch):
    """``LLMIO_LEVEL3_MODEL`` overrides the baked level3 model (must be a
    full combined identifier or bare name combined with baked prefix)."""
    set_env(
        monkeypatch,
        LLMIO_LEVEL1_TRANSPORT="claude-sdk",
        LLMIO_LEVEL1_MODEL="opus",
        LLMIO_LEVEL3_MODEL="haiku",
    )
    cfg = load_tier_config()
    # "haiku" is a bare model name → combined with baked prefix "claudeSDK"
    assert cfg.level3.model == "claudeSDK-haiku"
    assert cfg.level3.model_name == "haiku"


def test_env_provider_backward_compat(monkeypatch: pytest.MonkeyPatch) -> None:
    """``LLMIO_LEVEL{n}_PROVIDER`` is converted to a transport then combined
    with model."""
    set_env(
        monkeypatch,
        LLMIO_LEVEL1_PROVIDER="openrouter-deepseek",
        LLMIO_LEVEL1_MODEL="deepseek/deepseek-v4-flash",
    )
    cfg = load_tier_config()
    assert cfg.level1.model == "openrouter[deepseek]-deepseek/deepseek-v4-flash"
    assert cfg.level1.provider == "openrouter"


def test_env_transport_takes_precedence_over_provider(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``_TRANSPORT`` wins over ``_PROVIDER`` for the same level."""
    set_env(
        monkeypatch,
        LLMIO_LEVEL1_TRANSPORT="claude-sdk",
        LLMIO_LEVEL1_PROVIDER="openrouter-deepseek",
        LLMIO_LEVEL1_MODEL="opus",
    )
    cfg = load_tier_config()
    assert cfg.level1.model == "claudeSDK-opus"
    assert cfg.level1.provider == "claudeSDK"


# ========================================================================== #
#  Unknown prefix
# ========================================================================== #


def test_unknown_transport_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    """An unknown transport combined with a model creates an unknown prefix
    which surfaces as ``TierConfigLoadError``."""
    set_env(
        monkeypatch,
        LLMIO_LEVEL1_TRANSPORT="not-a-transport",
        LLMIO_LEVEL1_MODEL="opus",
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
        LLMIO_LEVEL1_TRANSPORT="claude-sdk",
        LLMIO_LEVEL1_MODEL="opus",
        LLMIO_LEVEL2_PROVIDER_KWARGS='{"timeout": 30, "base_url": "https://x"}',
    )
    cfg = load_tier_config()
    assert cfg.level2.provider_kwargs == {"timeout": 30, "base_url": "https://x"}


def test_env_provider_kwargs_empty_object(monkeypatch: pytest.MonkeyPatch):
    """``*_PROVIDER_KWARGS`` can be an empty JSON object."""
    set_env(
        monkeypatch,
        LLMIO_LEVEL1_TRANSPORT="claude-sdk",
        LLMIO_LEVEL1_MODEL="opus",
        LLMIO_LEVEL3_PROVIDER_KWARGS="{}",
    )
    cfg = load_tier_config()
    assert cfg.level3.provider_kwargs == {}


# ========================================================================== #
#  PROVIDER_KWARGS — invalid JSON
# ========================================================================== #


def test_env_provider_kwargs_invalid_json(monkeypatch: pytest.MonkeyPatch):
    """Invalid JSON in a ``*_PROVIDER_KWARGS`` var raises TierConfigLoadError."""
    set_env(
        monkeypatch,
        LLMIO_LEVEL1_TRANSPORT="claude-sdk",
        LLMIO_LEVEL1_MODEL="opus",
        LLMIO_LEVEL1_PROVIDER_KWARGS="not json",
    )
    with pytest.raises(TierConfigLoadError) as exc_info:
        load_tier_config()
    msg = str(exc_info.value)
    assert "LLMIO_LEVEL1_PROVIDER_KWARGS" in msg
    assert "not json" in msg or "Invalid JSON" in msg


def test_env_provider_kwargs_json_array(monkeypatch: pytest.MonkeyPatch):
    """A JSON array is not a valid value for provider_kwargs."""
    set_env(
        monkeypatch,
        LLMIO_LEVEL1_TRANSPORT="claude-sdk",
        LLMIO_LEVEL1_MODEL="opus",
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
        LLMIO_LEVEL1_TRANSPORT="claude-sdk",
        LLMIO_LEVEL1_MODEL="opus",
    )
    cfg = load_tier_config(
        {
            "level1": {
                "model": "openrouter[deepseek]-deepseek/deepseek-v4-pro",
            }
        }
    )
    assert cfg.level1.model == "openrouter[deepseek]-deepseek/deepseek-v4-pro"
    assert cfg.level1.provider == "openrouter"
    assert cfg.level1.model_name == "deepseek/deepseek-v4-pro"


def test_explicit_dict_partial_overrides_env(monkeypatch: pytest.MonkeyPatch):
    """A partial config_dict overrides only the specified fields."""
    set_env(
        monkeypatch,
        LLMIO_LEVEL1_TRANSPORT="claude-sdk",
        LLMIO_LEVEL1_MODEL="opus",
    )
    # Override only the model field with a full combined identifier.
    cfg = load_tier_config({"level1": {"model": "claudeSDK-haiku"}})
    assert cfg.level1.model == "claudeSDK-haiku"
    assert cfg.level1.model_name == "haiku"


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
        LLMIO_LEVEL1_TRANSPORT="claude-sdk",
        LLMIO_LEVEL1_MODEL="opus",
    )
    cfg = load_tier_config(
        {"level2": {"model": "openrouter[deepseek]-deepseek/deepseek-v4-flash"}}
    )
    assert cfg.level2.model == "openrouter[deepseek]-deepseek/deepseek-v4-flash"
    assert cfg.level2.model_name == "deepseek/deepseek-v4-flash"


# ========================================================================== #
#  Missing level1 entirely
# ========================================================================== #


def test_missing_level1_raises():
    """If level1 is missing from both env and dict, TierConfigLoadError is raised."""
    with pytest.raises(TierConfigLoadError) as exc_info:
        load_tier_config({})
    assert exc_info.value.__cause__ is not None


def test_missing_level1_partial_other_tiers(monkeypatch: pytest.MonkeyPatch):
    """Providing only level2/level3 without level1 still raises."""
    set_env(
        monkeypatch,
        LLMIO_LEVEL2_TRANSPORT="claude-sdk",
        LLMIO_LEVEL2_MODEL="opus",
    )
    with pytest.raises(TierConfigLoadError):
        load_tier_config({"level3": {"model": "claudeSDK-haiku"}})


# ========================================================================== #
#  Legacy LLMIO_FLASH_MODEL
# ========================================================================== #


def test_legacy_flash_model(monkeypatch: pytest.MonkeyPatch):
    """``LLMIO_FLASH_MODEL`` → level1.model + FutureWarning."""
    set_env(monkeypatch, LLMIO_FLASH_MODEL="haiku")
    # Must also provide a transport for level1.
    set_env(monkeypatch, LLMIO_LEVEL1_TRANSPORT="claude-sdk")

    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        cfg = load_tier_config()

    assert cfg.level1.model == "claudeSDK-haiku"
    assert cfg.level1.model_name == "haiku"
    assert len(w) == 1
    assert issubclass(w[0].category, FutureWarning)
    assert "LLMIO_FLASH_MODEL" in str(w[0].message)
    assert "LLMIO_LEVEL1_MODEL" in str(w[0].message)


def test_legacy_flash_model_suppressed_by_new(monkeypatch: pytest.MonkeyPatch):
    """Legacy var is ignored when the new-style var is also set."""
    set_env(
        monkeypatch,
        LLMIO_FLASH_MODEL="sonnet",
        LLMIO_LEVEL1_MODEL="opus",
        LLMIO_LEVEL1_TRANSPORT="claude-sdk",
    )
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        cfg = load_tier_config()

    assert cfg.level1.model == "claudeSDK-opus"
    # No FutureWarning for LLMIO_FLASH_MODEL
    flash_warnings = [x for x in w if "LLMIO_FLASH_MODEL" in str(x.message)]
    assert len(flash_warnings) == 0


# ========================================================================== #
#  Legacy LLMIO_FLASH_PROVIDER
# ========================================================================== #


def test_legacy_flash_provider(monkeypatch: pytest.MonkeyPatch):
    """``LLMIO_FLASH_PROVIDER`` → level1.transport + FutureWarning."""
    set_env(
        monkeypatch,
        LLMIO_FLASH_PROVIDER="claude-sdk",
        LLMIO_LEVEL1_MODEL="opus",
    )
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        cfg = load_tier_config()

    assert cfg.level1.model == "claudeSDK-opus"
    assert cfg.level1.provider == "claudeSDK"
    assert len(w) == 1
    assert issubclass(w[0].category, FutureWarning)
    assert "LLMIO_FLASH_PROVIDER" in str(w[0].message)


def test_legacy_flash_provider_suppressed_by_new(monkeypatch: pytest.MonkeyPatch):
    """Legacy flash provider is ignored when new ``_TRANSPORT`` is set."""
    set_env(
        monkeypatch,
        LLMIO_FLASH_PROVIDER="claude-sdk",
        LLMIO_LEVEL1_TRANSPORT="openrouter[deepseek]",
        LLMIO_LEVEL1_MODEL="deepseek/deepseek-v4-flash",
    )
    with warnings.catch_warnings(record=True):
        warnings.simplefilter("always")
        cfg = load_tier_config()

    assert cfg.level1.model == "openrouter[deepseek]-deepseek/deepseek-v4-flash"
    assert cfg.level1.provider == "openrouter"


# ========================================================================== #
#  Legacy LLMIO_NORMAL_MODEL
# ========================================================================== #


def test_legacy_normal_model(monkeypatch: pytest.MonkeyPatch):
    """``LLMIO_NORMAL_MODEL`` → level2.model + FutureWarning."""
    set_env(
        monkeypatch,
        LLMIO_NORMAL_MODEL="deepseek/deepseek-v4-flash",
        LLMIO_LEVEL1_TRANSPORT="claude-sdk",
        LLMIO_LEVEL1_MODEL="opus",
    )
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        cfg = load_tier_config()

    assert cfg.level2.model == "openrouter[deepseek]-deepseek/deepseek-v4-flash"
    assert cfg.level2.model_name == "deepseek/deepseek-v4-flash"
    assert len(w) == 1
    assert issubclass(w[0].category, FutureWarning)
    assert "LLMIO_NORMAL_MODEL" in str(w[0].message)


def test_legacy_normal_model_suppressed_by_new(monkeypatch: pytest.MonkeyPatch):
    """Legacy normal model is ignored when new var is set."""
    set_env(
        monkeypatch,
        LLMIO_NORMAL_MODEL="deepseek/deepseek-v4-pro",
        LLMIO_LEVEL2_MODEL="deepseek/deepseek-v4-flash",
        LLMIO_LEVEL1_TRANSPORT="claude-sdk",
        LLMIO_LEVEL1_MODEL="opus",
    )
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        cfg = load_tier_config()

    assert cfg.level2.model == "openrouter[deepseek]-deepseek/deepseek-v4-flash"
    normal_warnings = [x for x in w if "LLMIO_NORMAL_MODEL" in str(x.message)]
    assert len(normal_warnings) == 0


# ========================================================================== #
#  Legacy LLMIO_NORMAL_PROVIDER
# ========================================================================== #


def test_legacy_normal_provider(monkeypatch: pytest.MonkeyPatch):
    """``LLMIO_NORMAL_PROVIDER`` → level2.transport + FutureWarning."""
    set_env(
        monkeypatch,
        LLMIO_NORMAL_PROVIDER="claude-sdk",
        LLMIO_LEVEL1_TRANSPORT="claude-sdk",
        LLMIO_LEVEL1_MODEL="opus",
        # level2 model must be provided too (or baked default kicks in).
        LLMIO_LEVEL2_MODEL="opus",
    )
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        cfg = load_tier_config()

    assert cfg.level2.model == "claudeSDK-opus"
    normal_warnings = [x for x in w if "LLMIO_NORMAL_PROVIDER" in str(x.message)]
    assert len(normal_warnings) == 1
    assert issubclass(normal_warnings[0].category, FutureWarning)
    assert "LLMIO_NORMAL_PROVIDER" in str(normal_warnings[0].message)


def test_legacy_normal_provider_suppressed_by_new(monkeypatch: pytest.MonkeyPatch):
    """Legacy normal provider is ignored when new ``_TRANSPORT`` is set."""
    set_env(
        monkeypatch,
        LLMIO_NORMAL_PROVIDER="claude-sdk",
        LLMIO_LEVEL2_TRANSPORT="openrouter[deepseek]",
        LLMIO_LEVEL1_TRANSPORT="claude-sdk",
        LLMIO_LEVEL1_MODEL="opus",
    )
    with warnings.catch_warnings(record=True):
        warnings.simplefilter("always")
        cfg = load_tier_config()

    # Baked level2 model prefix is "openrouter[deepseek]" (from baked default);
    # transport from env overrides it → "openrouter[deepseek]-deepseek/deepseek-v4-pro"
    assert cfg.level2.model == "openrouter[deepseek]-deepseek/deepseek-v4-pro"
    assert cfg.level2.provider == "openrouter"


# ========================================================================== #
#  Legacy LLMIO_PROVIDER — fallback for all levels
# ========================================================================== #


def test_legacy_llmio_provider_fills_all_levels(monkeypatch: pytest.MonkeyPatch):
    """``LLMIO_PROVIDER`` applies to every level that lacks a transport."""
    set_env(
        monkeypatch,
        LLMIO_PROVIDER="claude-sdk",
        LLMIO_LEVEL1_MODEL="opus",
        LLMIO_LEVEL2_MODEL="haiku",
        LLMIO_LEVEL3_MODEL="sonnet",
    )
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        cfg = load_tier_config()

    assert cfg.level1.model == "claudeSDK-opus"
    assert cfg.level2.model == "claudeSDK-haiku"
    assert cfg.level3.model == "claudeSDK-sonnet"
    # One warning total for LLMIO_PROVIDER.
    provider_warnings = [x for x in w if "LLMIO_PROVIDER" in str(x.message)]
    assert len(provider_warnings) == 1
    assert issubclass(provider_warnings[0].category, FutureWarning)


def test_legacy_llmio_provider_respects_explicit_level1(
    monkeypatch: pytest.MonkeyPatch,
):
    """``LLMIO_PROVIDER`` does NOT override a level that already has a transport."""
    set_env(
        monkeypatch,
        LLMIO_PROVIDER="claude-sdk",
        LLMIO_LEVEL1_TRANSPORT="openrouter[deepseek]",
        LLMIO_LEVEL1_MODEL="deepseek/deepseek-v4-flash",
        LLMIO_LEVEL2_MODEL="haiku",
        LLMIO_LEVEL3_MODEL="sonnet",
    )
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        cfg = load_tier_config()

    assert cfg.level1.model == "openrouter[deepseek]-deepseek/deepseek-v4-flash"
    assert cfg.level2.model == "claudeSDK-haiku"
    assert cfg.level3.model == "claudeSDK-sonnet"
    # Still one warning.
    provider_warnings = [x for x in w if "LLMIO_PROVIDER" in str(x.message)]
    assert len(provider_warnings) == 1


def test_legacy_llmio_provider_respects_legacy_flash(monkeypatch: pytest.MonkeyPatch):
    """``LLMIO_PROVIDER`` does NOT override level1 when LLMIO_FLASH_PROVIDER is set."""
    set_env(
        monkeypatch,
        LLMIO_PROVIDER="claude-sdk",
        LLMIO_FLASH_PROVIDER="openrouter[deepseek]",
        LLMIO_LEVEL1_MODEL="deepseek/deepseek-v4-flash",
        LLMIO_LEVEL2_MODEL="haiku",
        LLMIO_LEVEL3_MODEL="sonnet",
    )
    with warnings.catch_warnings(record=True):
        warnings.simplefilter("always")
        cfg = load_tier_config()

    assert cfg.level1.model == "openrouter[deepseek]-deepseek/deepseek-v4-flash"
    assert cfg.level2.model == "claudeSDK-haiku"
    assert cfg.level3.model == "claudeSDK-sonnet"


def test_legacy_llmio_provider_no_warning_when_not_used(
    monkeypatch: pytest.MonkeyPatch,
):
    """No warning for LLMIO_PROVIDER if every level already has a transport."""
    set_env(
        monkeypatch,
        LLMIO_PROVIDER="claude-sdk",
        LLMIO_LEVEL1_TRANSPORT="claude-sdk",
        LLMIO_LEVEL1_MODEL="opus",
        LLMIO_LEVEL2_TRANSPORT="claude-sdk",
        LLMIO_LEVEL2_MODEL="haiku",
        LLMIO_LEVEL3_TRANSPORT="claude-sdk",
        LLMIO_LEVEL3_MODEL="sonnet",
    )
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        load_tier_config()

    provider_warnings = [x for x in w if "LLMIO_PROVIDER" in str(x.message)]
    assert len(provider_warnings) == 0


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
        LLMIO_LEVEL1_TRANSPORT="claude-sdk",
        LLMIO_LEVEL1_MODEL="opus",
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
        LLMIO_LEVEL1_TRANSPORT="claude-sdk",
        LLMIO_LEVEL1_MODEL="opus",
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
        LLMIO_LEVEL1_TRANSPORT="claude-sdk",
        LLMIO_LEVEL1_MODEL="opus",
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

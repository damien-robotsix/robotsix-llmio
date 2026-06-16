"""Tests for the tier configuration loader.

Covers:
- ``load_tier_config()`` with no arguments / no env vars
- Environment-variable overrides (new-style)
- ``*_PROVIDER_KWARGS`` parsing (valid and invalid JSON)
- Explicit ``config_dict`` overriding env vars
- Partial ``config_dict`` merge
- Missing ``level1`` → ``TierConfigLoadError``
- Legacy environment variables → ``FutureWarning`` + correct mapping
- Legacy variables silently ignored when new-style is set
- ``LLMIO_PROVIDER`` fallback to all levels lacking an explicit provider
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
    TierConfig,
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


def set_env(
    monkeypatch: pytest.MonkeyPatch, **kwargs: str
) -> None:
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
    cfg = load_tier_config({"level1": {"provider": "p", "model": "m"}})
    assert cfg.level1.provider == "p"
    assert cfg.level1.model == "m"
    assert cfg.level2 == LEVEL2_DEFAULT
    assert cfg.level3 == LEVEL3_DEFAULT


def test_explicit_dict_full():
    """All three tiers can be supplied via explicit dict."""
    cfg = load_tier_config(
        {
            "level1": {"provider": "a", "model": "1"},
            "level2": {"provider": "b", "model": "2"},
            "level3": {"provider": "c", "model": "3"},
        }
    )
    assert cfg.level1.provider == "a"
    assert cfg.level2.provider == "b"
    assert cfg.level3.provider == "c"


def test_explicit_dict_none_passed():
    """Passing ``config_dict=None`` is the same as omitting it."""
    # Still raises because level1 is missing.
    with pytest.raises(TierConfigLoadError):
        load_tier_config(None)


# ========================================================================== #
#  Environment variable overrides (new-style)
# ========================================================================== #


def test_env_level1_model_override(monkeypatch: pytest.MonkeyPatch):
    """``LLMIO_LEVEL1_MODEL`` populates level1.model."""
    set_env(
        monkeypatch,
        LLMIO_LEVEL1_PROVIDER="env-p",
        LLMIO_LEVEL1_MODEL="env-m",
    )
    cfg = load_tier_config()
    assert cfg.level1.provider == "env-p"
    assert cfg.level1.model == "env-m"


def test_env_level2_provider_override(monkeypatch: pytest.MonkeyPatch):
    """``LLMIO_LEVEL2_PROVIDER`` overrides the baked level2 provider."""
    set_env(
        monkeypatch,
        LLMIO_LEVEL1_PROVIDER="p",
        LLMIO_LEVEL1_MODEL="m",
        LLMIO_LEVEL2_PROVIDER="custom-level2",
    )
    cfg = load_tier_config()
    assert cfg.level1.provider == "p"
    assert cfg.level2.provider == "custom-level2"
    # model not set via env, so it comes from the baked base.
    assert cfg.level2.model == LEVEL2_DEFAULT.model


def test_env_level3_model_override(monkeypatch: pytest.MonkeyPatch):
    """``LLMIO_LEVEL3_MODEL`` overrides the baked level3 model."""
    set_env(
        monkeypatch,
        LLMIO_LEVEL1_PROVIDER="p",
        LLMIO_LEVEL1_MODEL="m",
        LLMIO_LEVEL3_MODEL="custom-level3-model",
    )
    cfg = load_tier_config()
    assert cfg.level3.model == "custom-level3-model"
    # provider not set via env → comes from baked base.
    assert cfg.level3.provider == LEVEL3_DEFAULT.provider


# ========================================================================== #
#  PROVIDER_KWARGS — valid JSON
# ========================================================================== #


def test_env_provider_kwargs_valid_json(monkeypatch: pytest.MonkeyPatch):
    """``*_PROVIDER_KWARGS`` with valid JSON is parsed into a dict."""
    set_env(
        monkeypatch,
        LLMIO_LEVEL1_PROVIDER="p",
        LLMIO_LEVEL1_MODEL="m",
        LLMIO_LEVEL2_PROVIDER_KWARGS='{"timeout": 30, "base_url": "https://x"}',
    )
    cfg = load_tier_config()
    assert cfg.level2.provider_kwargs == {"timeout": 30, "base_url": "https://x"}


def test_env_provider_kwargs_empty_object(monkeypatch: pytest.MonkeyPatch):
    """``*_PROVIDER_KWARGS`` can be an empty JSON object."""
    set_env(
        monkeypatch,
        LLMIO_LEVEL1_PROVIDER="p",
        LLMIO_LEVEL1_MODEL="m",
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
        LLMIO_LEVEL1_PROVIDER="p",
        LLMIO_LEVEL1_MODEL="m",
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
        LLMIO_LEVEL1_PROVIDER="p",
        LLMIO_LEVEL1_MODEL="m",
        LLMIO_LEVEL3_PROVIDER_KWARGS='[1, 2, 3]',
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
        LLMIO_LEVEL1_PROVIDER="env-p",
        LLMIO_LEVEL1_MODEL="env-m",
    )
    cfg = load_tier_config(
        {"level1": {"provider": "dict-p", "model": "dict-m"}}
    )
    assert cfg.level1.provider == "dict-p"
    assert cfg.level1.model == "dict-m"


def test_explicit_dict_partial_overrides_env(monkeypatch: pytest.MonkeyPatch):
    """A partial config_dict overrides only the specified fields."""
    set_env(
        monkeypatch,
        LLMIO_LEVEL1_PROVIDER="env-p",
        LLMIO_LEVEL1_MODEL="env-m",
    )
    cfg = load_tier_config({"level1": {"model": "dict-m"}})
    assert cfg.level1.provider == "env-p"  # from env
    assert cfg.level1.model == "dict-m"  # overridden


# ========================================================================== #
#  Partial config_dict — merge correctness
# ========================================================================== #


def test_partial_dict_only_level1():
    """Only level1 in dict → level2/3 come from baked defaults."""
    cfg = load_tier_config({"level1": {"provider": "x", "model": "y"}})
    assert cfg.level2 == LEVEL2_DEFAULT
    assert cfg.level3 == LEVEL3_DEFAULT


def test_partial_dict_only_level2_model(monkeypatch: pytest.MonkeyPatch):
    """Setting only level2.model in dict works (baked provider is used)."""
    set_env(
        monkeypatch,
        LLMIO_LEVEL1_PROVIDER="p",
        LLMIO_LEVEL1_MODEL="m",
    )
    cfg = load_tier_config({"level2": {"model": "custom-model"}})
    assert cfg.level2.model == "custom-model"
    assert cfg.level2.provider == LEVEL2_DEFAULT.provider


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
        LLMIO_LEVEL2_PROVIDER="p2",
        LLMIO_LEVEL2_MODEL="m2",
    )
    with pytest.raises(TierConfigLoadError):
        load_tier_config({"level3": {"provider": "p3", "model": "m3"}})


# ========================================================================== #
#  Legacy LLMIO_FLASH_MODEL
# ========================================================================== #


def test_legacy_flash_model(monkeypatch: pytest.MonkeyPatch):
    """``LLMIO_FLASH_MODEL`` → level1.model + FutureWarning."""
    set_env(monkeypatch, LLMIO_FLASH_MODEL="flash-m")
    # Must also provide provider for level1.
    set_env(monkeypatch, LLMIO_LEVEL1_PROVIDER="p")

    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        cfg = load_tier_config()

    assert cfg.level1.model == "flash-m"
    assert len(w) == 1
    assert issubclass(w[0].category, FutureWarning)
    assert "LLMIO_FLASH_MODEL" in str(w[0].message)
    assert "LLMIO_LEVEL1_MODEL" in str(w[0].message)


def test_legacy_flash_model_suppressed_by_new(monkeypatch: pytest.MonkeyPatch):
    """Legacy var is ignored when the new-style var is also set."""
    set_env(
        monkeypatch,
        LLMIO_FLASH_MODEL="flash-m",
        LLMIO_LEVEL1_MODEL="new-m",
        LLMIO_LEVEL1_PROVIDER="p",
    )
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        cfg = load_tier_config()

    assert cfg.level1.model == "new-m"
    # No FutureWarning for LLMIO_FLASH_MODEL
    flash_warnings = [x for x in w if "LLMIO_FLASH_MODEL" in str(x.message)]
    assert len(flash_warnings) == 0


# ========================================================================== #
#  Legacy LLMIO_FLASH_PROVIDER
# ========================================================================== #


def test_legacy_flash_provider(monkeypatch: pytest.MonkeyPatch):
    """``LLMIO_FLASH_PROVIDER`` → level1.provider + FutureWarning."""
    set_env(
        monkeypatch,
        LLMIO_FLASH_PROVIDER="flash-p",
        LLMIO_LEVEL1_MODEL="m",
    )
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        cfg = load_tier_config()

    assert cfg.level1.provider == "flash-p"
    assert len(w) == 1
    assert issubclass(w[0].category, FutureWarning)
    assert "LLMIO_FLASH_PROVIDER" in str(w[0].message)


def test_legacy_flash_provider_suppressed_by_new(monkeypatch: pytest.MonkeyPatch):
    """Legacy flash provider is ignored when new var is set."""
    set_env(
        monkeypatch,
        LLMIO_FLASH_PROVIDER="flash-p",
        LLMIO_LEVEL1_PROVIDER="new-p",
        LLMIO_LEVEL1_MODEL="m",
    )
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        cfg = load_tier_config()

    assert cfg.level1.provider == "new-p"
    flash_warnings = [x for x in w if "LLMIO_FLASH_PROVIDER" in str(x.message)]
    assert len(flash_warnings) == 0


# ========================================================================== #
#  Legacy LLMIO_NORMAL_MODEL
# ========================================================================== #


def test_legacy_normal_model(monkeypatch: pytest.MonkeyPatch):
    """``LLMIO_NORMAL_MODEL`` → level2.model + FutureWarning."""
    set_env(
        monkeypatch,
        LLMIO_NORMAL_MODEL="normal-m",
        LLMIO_LEVEL1_PROVIDER="p",
        LLMIO_LEVEL1_MODEL="m",
    )
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        cfg = load_tier_config()

    assert cfg.level2.model == "normal-m"
    assert len(w) == 1
    assert issubclass(w[0].category, FutureWarning)
    assert "LLMIO_NORMAL_MODEL" in str(w[0].message)


def test_legacy_normal_model_suppressed_by_new(monkeypatch: pytest.MonkeyPatch):
    """Legacy normal model is ignored when new var is set."""
    set_env(
        monkeypatch,
        LLMIO_NORMAL_MODEL="normal-m",
        LLMIO_LEVEL2_MODEL="new-m",
        LLMIO_LEVEL1_PROVIDER="p",
        LLMIO_LEVEL1_MODEL="m",
    )
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        cfg = load_tier_config()

    assert cfg.level2.model == "new-m"
    normal_warnings = [x for x in w if "LLMIO_NORMAL_MODEL" in str(x.message)]
    assert len(normal_warnings) == 0


# ========================================================================== #
#  Legacy LLMIO_NORMAL_PROVIDER
# ========================================================================== #


def test_legacy_normal_provider(monkeypatch: pytest.MonkeyPatch):
    """``LLMIO_NORMAL_PROVIDER`` → level2.provider + FutureWarning."""
    set_env(
        monkeypatch,
        LLMIO_NORMAL_PROVIDER="normal-p",
        LLMIO_LEVEL1_PROVIDER="p",
        LLMIO_LEVEL1_MODEL="m",
    )
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        cfg = load_tier_config()

    assert cfg.level2.provider == "normal-p"
    assert len(w) == 1
    assert issubclass(w[0].category, FutureWarning)
    assert "LLMIO_NORMAL_PROVIDER" in str(w[0].message)


def test_legacy_normal_provider_suppressed_by_new(monkeypatch: pytest.MonkeyPatch):
    """Legacy normal provider is ignored when new var is set."""
    set_env(
        monkeypatch,
        LLMIO_NORMAL_PROVIDER="normal-p",
        LLMIO_LEVEL2_PROVIDER="new-p",
        LLMIO_LEVEL1_PROVIDER="p",
        LLMIO_LEVEL1_MODEL="m",
    )
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        cfg = load_tier_config()

    assert cfg.level2.provider == "new-p"
    normal_warnings = [x for x in w if "LLMIO_NORMAL_PROVIDER" in str(x.message)]
    assert len(normal_warnings) == 0


# ========================================================================== #
#  Legacy LLMIO_PROVIDER — fallback for all levels
# ========================================================================== #


def test_legacy_llmio_provider_fills_all_levels(monkeypatch: pytest.MonkeyPatch):
    """``LLMIO_PROVIDER`` applies to every level that lacks a provider."""
    set_env(
        monkeypatch,
        LLMIO_PROVIDER="global-p",
        LLMIO_LEVEL1_MODEL="m1",
        LLMIO_LEVEL2_MODEL="m2",
        LLMIO_LEVEL3_MODEL="m3",
    )
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        cfg = load_tier_config()

    assert cfg.level1.provider == "global-p"
    assert cfg.level2.provider == "global-p"
    assert cfg.level3.provider == "global-p"
    # One warning total for LLMIO_PROVIDER.
    provider_warnings = [x for x in w if "LLMIO_PROVIDER" in str(x.message)]
    assert len(provider_warnings) == 1
    assert issubclass(provider_warnings[0].category, FutureWarning)


def test_legacy_llmio_provider_respects_explicit_level1(monkeypatch: pytest.MonkeyPatch):
    """``LLMIO_PROVIDER`` does NOT override a level that already has a provider."""
    set_env(
        monkeypatch,
        LLMIO_PROVIDER="global-p",
        LLMIO_LEVEL1_PROVIDER="explicit-p1",
        LLMIO_LEVEL1_MODEL="m1",
        LLMIO_LEVEL2_MODEL="m2",
        LLMIO_LEVEL3_MODEL="m3",
    )
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        cfg = load_tier_config()

    assert cfg.level1.provider == "explicit-p1"
    assert cfg.level2.provider == "global-p"
    assert cfg.level3.provider == "global-p"
    # Still one warning.
    provider_warnings = [x for x in w if "LLMIO_PROVIDER" in str(x.message)]
    assert len(provider_warnings) == 1


def test_legacy_llmio_provider_respects_legacy_flash(monkeypatch: pytest.MonkeyPatch):
    """``LLMIO_PROVIDER`` does NOT override level1 when LLMIO_FLASH_PROVIDER is set."""
    set_env(
        monkeypatch,
        LLMIO_PROVIDER="global-p",
        LLMIO_FLASH_PROVIDER="flash-p",
        LLMIO_LEVEL1_MODEL="m1",
        LLMIO_LEVEL2_MODEL="m2",
        LLMIO_LEVEL3_MODEL="m3",
    )
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        cfg = load_tier_config()

    assert cfg.level1.provider == "flash-p"
    assert cfg.level2.provider == "global-p"
    assert cfg.level3.provider == "global-p"


def test_legacy_llmio_provider_no_warning_when_not_used(
    monkeypatch: pytest.MonkeyPatch,
):
    """No warning for LLMIO_PROVIDER if every level already has a provider."""
    set_env(
        monkeypatch,
        LLMIO_PROVIDER="global-p",
        LLMIO_LEVEL1_PROVIDER="p1",
        LLMIO_LEVEL1_MODEL="m1",
        LLMIO_LEVEL2_PROVIDER="p2",
        LLMIO_LEVEL2_MODEL="m2",
        LLMIO_LEVEL3_PROVIDER="p3",
        LLMIO_LEVEL3_MODEL="m3",
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

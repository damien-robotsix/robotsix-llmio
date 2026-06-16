"""Tests for PEP 562 __getattr__ lazy-import branches in config/__init__.py.

Every public name in the config package's ``__all__`` must resolve
when imported from ``robotsix_llmio.config`` — this ensures all
``__getattr__`` branches are covered.
"""

from __future__ import annotations

import pytest

from robotsix_llmio.config import tier as _tier
from robotsix_llmio.config import loader as _loader


def test_re_export_legacy_tier_map():
    from robotsix_llmio.config import LEGACY_TIER_MAP

    assert LEGACY_TIER_MAP is _tier.LEGACY_TIER_MAP


def test_re_export_level1_default():
    from robotsix_llmio.config import LEVEL1_DEFAULT

    assert LEVEL1_DEFAULT is _tier.LEVEL1_DEFAULT


def test_re_export_level2_default():
    from robotsix_llmio.config import LEVEL2_DEFAULT

    assert LEVEL2_DEFAULT is _tier.LEVEL2_DEFAULT


def test_re_export_level3_default():
    from robotsix_llmio.config import LEVEL3_DEFAULT

    assert LEVEL3_DEFAULT is _tier.LEVEL3_DEFAULT


def test_re_export_tier_config():
    from robotsix_llmio.config import TierConfig

    assert TierConfig is _tier.TierConfig


def test_re_export_tier_level():
    from robotsix_llmio.config import TierLevel

    assert TierLevel is _tier.TierLevel


def test_re_export_tier_level_config():
    from robotsix_llmio.config import TierLevelConfig

    assert TierLevelConfig is _tier.TierLevelConfig


def test_re_export_tier_config_load_error():
    from robotsix_llmio.config import TierConfigLoadError

    assert TierConfigLoadError is _loader.TierConfigLoadError


def test_re_export_load_tier_config():
    from robotsix_llmio.config import load_tier_config

    assert load_tier_config is _loader.load_tier_config


@pytest.mark.parametrize(
    "attr_name",
    [
        "ModelWeightConfig",
        "WeeklyPaceConfig",
    ],
)
def test_re_export_weekly_pace_names(attr_name: str):
    """Every weekly-pace name in __all__ resolves via __getattr__."""
    import robotsix_llmio.config as _pkg

    obj = getattr(_pkg, attr_name)
    assert obj is not None


def test_unknown_attribute_raises():
    """Accessing a non-existent name raises AttributeError."""
    import robotsix_llmio.config as _pkg

    with pytest.raises(AttributeError):
        _pkg._nonexistent_name  # noqa: SLF001

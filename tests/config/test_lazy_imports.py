"""Tests for PEP 562 __getattr__ lazy-import branches in config/__init__.py.

Every public name in the config package's ``__all__`` must resolve
when imported from ``robotsix_llmio.config`` — this ensures all
``__getattr__`` branches are covered.
"""

from __future__ import annotations

import pytest

from robotsix_llmio.config import loader as _loader
from robotsix_llmio.config import tier as _tier


def test_re_export_legacy_tier_map():
    with pytest.warns(DeprecationWarning, match="LEGACY_TIER_MAP is deprecated"):
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


def test_re_export_unknown_transport_error():
    from robotsix_llmio.config import UnknownTransportError
    from robotsix_llmio.config import transport as _transport

    assert UnknownTransportError is _transport.UnknownTransportError


def test_re_export_validate_transport():
    from robotsix_llmio.config import transport as _transport
    from robotsix_llmio.config import validate_transport

    assert validate_transport is _transport.validate_transport


@pytest.mark.parametrize(
    "attr_name",
    [
        "MODEL_LEVEL_TO_TIER",
        "TRANSPORT_ALIASES",
        "UnknownTransportError",
        "MalformedIdentifierError",
        "ParsedIdentifier",
        "ModelWeightConfig",
        "WeeklyPaceConfig",
        "create_model",
        "get_provider_for_identifier",
        "parse_model_identifier",
        "validate_transport",
    ],
)
def test_re_export_names(attr_name: str):
    """Every name in __all__ resolves via __getattr__."""
    import robotsix_llmio.config as _pkg

    obj = getattr(_pkg, attr_name)
    assert obj is not None


def test_unknown_attribute_raises():
    """Accessing a non-existent name raises AttributeError."""
    import robotsix_llmio.config as _pkg

    with pytest.raises(AttributeError):
        _ = _pkg._nonexistent_name

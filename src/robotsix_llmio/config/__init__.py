"""Three-tier configuration schema for provider+model bindings.

Prefer importing from :mod:`robotsix_llmio.core` — the public types are
re-exported there for discoverability.

Imports are deferred via :pep:`562` ``__getattr__`` so that importing the
config package does not eagerly pull in pydantic until a name is accessed.
"""

from __future__ import annotations

import warnings
from typing import Any

__all__ = [
    "LEGACY_TIER_MAP",
    "LEVEL1_DEFAULT",
    "LEVEL2_DEFAULT",
    "LEVEL3_DEFAULT",
    "MODEL_LEVEL_TO_TIER",
    "PROVIDER_MODELS",
    "TRANSPORT_ALIASES",
    "ModelWeightConfig",
    "TierConfig",
    "TierConfigLoadError",
    "TierLevel",
    "TierLevelConfig",
    "UnknownModelError",
    "WeeklyPaceConfig",
    "create_model",
    "load_tier_config",
    "validate_model",
]


def __getattr__(name: str) -> Any:  # PEP 562 — lazy imports
    if name == "LEGACY_TIER_MAP":
        warnings.warn(
            "LEGACY_TIER_MAP is deprecated. Use TierConfig.for_level() instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        from . import tier

        return tier.LEGACY_TIER_MAP
    if name == "LEVEL1_DEFAULT":
        from . import tier

        return tier.LEVEL1_DEFAULT
    if name == "LEVEL2_DEFAULT":
        from . import tier

        return tier.LEVEL2_DEFAULT
    if name == "LEVEL3_DEFAULT":
        from . import tier

        return tier.LEVEL3_DEFAULT
    if name == "TierConfig":
        from . import tier

        return tier.TierConfig
    if name == "TierLevel":
        from . import tier

        return tier.TierLevel
    if name == "TierLevelConfig":
        from . import tier

        return tier.TierLevelConfig
    if name == "TierConfigLoadError":
        from . import loader

        return loader.TierConfigLoadError
    if name == "load_tier_config":
        from . import loader

        return loader.load_tier_config
    if name == "ModelWeightConfig":
        from . import weekly_pace

        return weekly_pace.ModelWeightConfig
    if name == "WeeklyPaceConfig":
        from . import weekly_pace

        return weekly_pace.WeeklyPaceConfig
    if name == "create_model":
        from . import factory

        return factory.create_model
    if name == "TRANSPORT_ALIASES":
        from . import transport

        return transport.TRANSPORT_ALIASES
    if name == "MODEL_LEVEL_TO_TIER":
        from . import transport

        return transport.MODEL_LEVEL_TO_TIER
    if name == "PROVIDER_MODELS":
        from . import model_registry

        return model_registry.PROVIDER_MODELS
    if name == "UnknownModelError":
        from . import model_registry

        return model_registry.UnknownModelError
    if name == "validate_model":
        from . import model_registry

        return model_registry.validate_model
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

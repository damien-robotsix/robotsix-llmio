"""Three-tier configuration schema for provider+model bindings.

Prefer importing from :mod:`robotsix_llmio.core` — the public types are
re-exported there for discoverability.

Imports are deferred via :pep:`562` ``__getattr__`` so that importing the
config package does not eagerly pull in pydantic until a name is accessed.
"""

from __future__ import annotations

from typing import Any

__all__ = [
    "LEGACY_TIER_MAP",
    "LEVEL1_DEFAULT",
    "LEVEL2_DEFAULT",
    "LEVEL3_DEFAULT",
    "ModelWeightConfig",
    "TierConfig",
    "TierConfigLoadError",
    "TierLevel",
    "TierLevelConfig",
    "WeeklyPaceConfig",
    "load_tier_config",
]


def __getattr__(name: str) -> Any:  # PEP 562 — lazy imports
    if name == "LEGACY_TIER_MAP":
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
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

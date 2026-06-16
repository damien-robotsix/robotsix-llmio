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
    "TierLevel",
    "TierLevelConfig",
    "WeeklyPaceConfig",
]

_TIER_NAMES: frozenset[str] = frozenset(
    {
        "LEGACY_TIER_MAP",
        "LEVEL1_DEFAULT",
        "LEVEL2_DEFAULT",
        "LEVEL3_DEFAULT",
        "TierConfig",
        "TierLevel",
        "TierLevelConfig",
    }
)

_WEEKLY_PACE_NAMES: frozenset[str] = frozenset(
    {"ModelWeightConfig", "WeeklyPaceConfig"}
)


def __getattr__(name: str) -> Any:  # PEP 562 — lazy imports
    if name in _TIER_NAMES:
        from . import tier  # intentional lazy import (PEP 562)

        if name == "LEGACY_TIER_MAP":
            return tier.LEGACY_TIER_MAP
        if name == "LEVEL1_DEFAULT":
            return tier.LEVEL1_DEFAULT
        if name == "LEVEL2_DEFAULT":
            return tier.LEVEL2_DEFAULT
        if name == "LEVEL3_DEFAULT":
            return tier.LEVEL3_DEFAULT
        if name == "TierConfig":
            return tier.TierConfig
        if name == "TierLevel":
            return tier.TierLevel
        if name == "TierLevelConfig":
            return tier.TierLevelConfig
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    if name in _WEEKLY_PACE_NAMES:
        from . import weekly_pace  # intentional lazy import (PEP 562)

        if name == "ModelWeightConfig":
            return weekly_pace.ModelWeightConfig
        if name == "WeeklyPaceConfig":
            return weekly_pace.WeeklyPaceConfig
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

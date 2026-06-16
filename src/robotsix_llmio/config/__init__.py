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

        return getattr(tier, name)
    if name in _WEEKLY_PACE_NAMES:
        from . import weekly_pace  # intentional lazy import (PEP 562)

        return getattr(weekly_pace, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

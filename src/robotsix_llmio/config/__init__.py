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

_MODULES: dict[str, str] = {
    "LEGACY_TIER_MAP": "tier",
    "LEVEL1_DEFAULT": "tier",
    "LEVEL2_DEFAULT": "tier",
    "LEVEL3_DEFAULT": "tier",
    "ModelWeightConfig": "weekly_pace",
    "TierConfig": "tier",
    "TierLevel": "tier",
    "TierLevelConfig": "tier",
    "WeeklyPaceConfig": "weekly_pace",
}


def __getattr__(name: str) -> Any:  # PEP 562 — lazy imports
    if name in _MODULES:
        from importlib import import_module

        mod = import_module(f".{_MODULES[name]}", __package__)
        return getattr(mod, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

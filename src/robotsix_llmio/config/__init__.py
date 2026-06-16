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
    "TierConfig",
    "TierLevel",
    "TierLevelConfig",
]


def __getattr__(name: str) -> Any:  # PEP 562 — lazy imports
    if name in __all__:
        from . import tier

        return getattr(tier, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

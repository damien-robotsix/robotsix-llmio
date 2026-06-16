"""Three-tier configuration schema for provider+model bindings.

Prefer importing from :mod:`robotsix_llmio.core` — the public types are
re-exported there for discoverability.
"""

from __future__ import annotations

from .tier import (
    LEGACY_TIER_MAP,
    LEVEL1_DEFAULT,
    LEVEL2_DEFAULT,
    LEVEL3_DEFAULT,
    TierConfig,
    TierLevel,
    TierLevelConfig,
)

__all__ = [
    "LEGACY_TIER_MAP",
    "LEVEL1_DEFAULT",
    "LEVEL2_DEFAULT",
    "LEVEL3_DEFAULT",
    "TierConfig",
    "TierLevel",
    "TierLevelConfig",
]

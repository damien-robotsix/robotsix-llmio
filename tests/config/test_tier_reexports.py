"""Re-export identity checks — verify ``robotsix_llmio.core`` re-exports match
the canonical tier symbols."""

from __future__ import annotations

from robotsix_llmio.config.tier import (
    LEVEL1_DEFAULT,
    LEVEL2_DEFAULT,
    LEVEL3_DEFAULT,
    LEVEL4_DEFAULT,
    TierConfig,
    TierLevel,
    TierLevelConfig,
)


def test_core_reexports_tier_level():
    """``TierLevel`` is importable from ``robotsix_llmio.core``."""
    from robotsix_llmio.core import TierLevel as TL

    assert TL is TierLevel


def test_core_reexports_tier_config():
    """``TierConfig`` is importable from ``robotsix_llmio.core``."""
    from robotsix_llmio.core import TierConfig as TC

    assert TC is TierConfig


def test_core_reexports_tier_level_config():
    """``TierLevelConfig`` is importable from ``robotsix_llmio.core``."""
    from robotsix_llmio.core import TierLevelConfig as TLC

    assert TLC is TierLevelConfig


def test_core_reexports_defaults():
    """The four baked defaults are importable from ``robotsix_llmio.core``."""
    from robotsix_llmio.core import (
        LEVEL1_DEFAULT as L1D,
    )
    from robotsix_llmio.core import (
        LEVEL2_DEFAULT as L2D,
    )
    from robotsix_llmio.core import (
        LEVEL3_DEFAULT as L3D,
    )
    from robotsix_llmio.core import (
        LEVEL4_DEFAULT as L4D,
    )

    assert L1D is LEVEL1_DEFAULT
    assert L2D is LEVEL2_DEFAULT
    assert L3D is LEVEL3_DEFAULT
    assert L4D is LEVEL4_DEFAULT

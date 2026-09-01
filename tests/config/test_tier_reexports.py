"""Re-export identity checks — verify ``robotsix_llmio.core`` re-exports match
the canonical tier symbols."""

from __future__ import annotations

from robotsix_llmio.config.tier import (
    DEFAULT_LEVEL1,
    DEFAULT_LEVEL2,
    DEFAULT_LEVEL3,
    FALLBACK_LEVEL1,
    FALLBACK_LEVEL2,
    FALLBACK_LEVEL3,
    FailoverConfig,
    ProviderSlotConfig,
    TierConfig,
    TierLevel,
    TierLevelConfig,
)


def test_core_reexports_schema_types():
    from robotsix_llmio.core import (
        FailoverConfig as FC,
    )
    from robotsix_llmio.core import (
        ProviderSlotConfig as PSC,
    )
    from robotsix_llmio.core import (
        TierConfig as TC,
    )
    from robotsix_llmio.core import (
        TierLevel as TL,
    )
    from robotsix_llmio.core import (
        TierLevelConfig as TLC,
    )

    assert TC is TierConfig
    assert TL is TierLevel
    assert TLC is TierLevelConfig
    assert PSC is ProviderSlotConfig
    assert FC is FailoverConfig


def test_core_reexports_baked_defaults():
    from robotsix_llmio import core

    assert core.DEFAULT_LEVEL1 is DEFAULT_LEVEL1
    assert core.DEFAULT_LEVEL2 is DEFAULT_LEVEL2
    assert core.DEFAULT_LEVEL3 is DEFAULT_LEVEL3
    assert core.FALLBACK_LEVEL1 is FALLBACK_LEVEL1
    assert core.FALLBACK_LEVEL2 is FALLBACK_LEVEL2
    assert core.FALLBACK_LEVEL3 is FALLBACK_LEVEL3


def test_core_reexports_failover_surface():
    from robotsix_llmio.core import (
        FailoverStatus,
        acall_with_failover,
        call_with_failover,
        get_failover_status,
        get_failover_tracker,
    )
    from robotsix_llmio.core import failover as failover_module

    assert FailoverStatus is failover_module.FailoverStatus
    assert call_with_failover is failover_module.call_with_failover
    assert acall_with_failover is failover_module.acall_with_failover
    assert get_failover_status is failover_module.get_failover_status
    assert get_failover_tracker is failover_module.get_failover_tracker

"""Tier-fallback: default_factory deep-copy isolation regression.

The sync and async ``call_with_tier_fallback`` / ``acall_with_tier_fallback``
tests have been split into:

* ``test_core_tier_fallback_sync.py``
* ``test_core_tier_fallback_async.py``

This file retains only the deep-copy isolation regression test.
"""

from __future__ import annotations

from robotsix_llmio.config.tier import (
    LEVEL1_DEFAULT,
    LEVEL2_DEFAULT,
    LEVEL3_DEFAULT,
    LEVEL4_DEFAULT,
    TierConfig,
)
from robotsix_llmio.core.factory import default_tier_config

# --------------------------------------------------------------------------- #
#  default_factory deep-copy isolation regression                             #
# --------------------------------------------------------------------------- #


def test_tier_config_default_instances_are_independent():
    """Regression: default_factory deep-copies prevent shared mutable state.

    Before the fix, ``TierConfig().level1`` was the *same object* as
    ``LEVEL1_DEFAULT`` (and as every other ``TierConfig().level1``), so
    mutating ``.provider_kwargs`` on one instance's slot corrupted all
    others.  This test asserts the fix.
    """

    # 1. Instance isolation: every TierConfig() gets independent slot copies.
    for attr in ("level1", "level2", "level3", "level4"):
        a = getattr(TierConfig(), attr)
        b = getattr(TierConfig(), attr)
        assert a is not b, f"TierConfig().{attr} is TierConfig().{attr}"

    # 2. Singleton isolation: no TierConfig() slot aliases the module-level
    #    baked singleton.
    for attr, singleton in (
        ("level1", LEVEL1_DEFAULT),
        ("level2", LEVEL2_DEFAULT),
        ("level3", LEVEL3_DEFAULT),
        ("level4", LEVEL4_DEFAULT),
    ):
        cfg_slot = getattr(TierConfig(), attr)
        assert cfg_slot is not singleton, (
            f"TierConfig().{attr} is LEVEL*_DEFAULT singleton"
        )

    # 3. Mutation does not bleed.
    for attr, singleton in (
        ("level1", LEVEL1_DEFAULT),
        ("level2", LEVEL2_DEFAULT),
        ("level3", LEVEL3_DEFAULT),
        ("level4", LEVEL4_DEFAULT),
    ):
        cfg1 = TierConfig()
        slot1 = getattr(cfg1, attr)
        slot1.provider_kwargs["api_key"] = "tenant-A"

        cfg2 = TierConfig()
        slot2 = getattr(cfg2, attr)
        assert "api_key" not in slot2.provider_kwargs, (
            f"mutation bled into fresh TierConfig().{attr}"
        )
        assert "api_key" not in singleton.provider_kwargs, (
            f"mutation bled into {attr.upper()}_DEFAULT singleton"
        )

    # 4. default_tier_config() isolation.
    for attr in ("level1", "level2", "level3", "level4"):
        a = getattr(default_tier_config(), attr)
        b = getattr(default_tier_config(), attr)
        assert a is not b, (
            f"default_tier_config().{attr} is default_tier_config().{attr}"
        )

    for attr, singleton in (
        ("level1", LEVEL1_DEFAULT),
        ("level2", LEVEL2_DEFAULT),
        ("level3", LEVEL3_DEFAULT),
        ("level4", LEVEL4_DEFAULT),
    ):
        cfg_slot = getattr(default_tier_config(), attr)
        assert cfg_slot is not singleton, (
            f"default_tier_config().{attr} is LEVEL*_DEFAULT singleton"
        )

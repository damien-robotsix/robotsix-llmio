"""Transport alias mappings — consumer-facing names for provider backends.

Consumers set ``transport: claude-sdk`` or ``transport: openrouter[deepseek]``
in their config and never name a concrete provider class.  This module maps
those aliases to the provider registry names consumed by
:func:`~robotsix_llmio.core.factory.get_provider`.
"""

from __future__ import annotations

from ..core.provider import Tier

# --------------------------------------------------------------------------- #
#  Transport aliases                                                          #
# --------------------------------------------------------------------------- #

TRANSPORT_ALIASES: dict[str, str] = {
    "claude-sdk": "claude-sdk",
    "openrouter[deepseek]": "openrouter-deepseek",
}
"""Consumer-facing transport name → provider registry name.

Add a new entry here when a new provider backend is registered.
"""

# --------------------------------------------------------------------------- #
#  Model level → Tier mapping                                                 #
# --------------------------------------------------------------------------- #

MODEL_LEVEL_TO_TIER: dict[int, Tier] = {
    1: Tier.CHEAP,
    2: Tier.DEFAULT,
    3: Tier.DEFAULT,
}
"""Map a consumer *model_level* (1, 2, or 3) to a :class:`Tier`.

**Deprecated** — prefer :meth:`TierConfig.for_level` which resolves
directly to a :class:`~.tier.TierLevelConfig` without the two-tier
round-trip.  This mapping is kept only for ``create_model()`` backward
compatibility and will be removed in a follow-up ticket.

Level 1 (cheap, repetitive tasks) maps to :attr:`Tier.CHEAP`; levels 2 and 3
both map to :attr:`Tier.DEFAULT`.  The three-tier :class:`~.tier.TierConfig`
schema reserves level 3 for high-level planning with a different provider
family; the legacy two-tier :class:`Tier` enum has no third tier, so level 3
falls back to ``DEFAULT``.
"""

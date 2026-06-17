"""Transport alias mappings — consumer-facing names for provider backends.

Consumers set ``transport: claude-sdk`` or ``transport: openrouter[deepseek]``
in their config and never name a concrete provider class.  This module maps
those aliases to the provider registry names consumed by
:func:`~robotsix_llmio.core.factory.get_provider`.
"""

from __future__ import annotations

from ..core.provider import Tier
from ..exceptions import RobotsixLLMIOError

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
#  Transport validation                                                       #
# --------------------------------------------------------------------------- #


class UnknownTransportError(RobotsixLLMIOError):
    """Raised when a transport alias is not recognised."""


def validate_transport(transport: str) -> None:
    """Check that *transport* is a known consumer-facing transport alias.

    Args:
        transport: Transport alias (e.g. ``"claude-sdk"`` or
            ``"openrouter[deepseek]"``).

    Raises:
        UnknownTransportError: If *transport* is not a key of
            :data:`TRANSPORT_ALIASES`.  The message names the bad value and
            lists the known transports.
    """
    if transport not in TRANSPORT_ALIASES:
        raise UnknownTransportError(
            f"Unknown transport {transport!r}. "
            f"Known transports: {', '.join(sorted(TRANSPORT_ALIASES))}."
        )


def provider_to_transport(value: str) -> str:
    """Convert a legacy provider registry name to a transport alias.

    Backward-compat converter used by the schema and loader to accept the
    old ``provider`` shape:

    * If *value* is already a transport alias (a key of
      :data:`TRANSPORT_ALIASES`), return it unchanged.  Note ``"claude-sdk"``
      is both an alias and a registry name; the alias key wins.
    * Else reverse-look-up :data:`TRANSPORT_ALIASES` (provider registry
      name → alias) and return the alias.
    * If *value* matches neither, return it unchanged so a downstream
      :func:`validate_transport` raises a clear error.
    """
    if value in TRANSPORT_ALIASES:
        return value
    for alias, provider in TRANSPORT_ALIASES.items():
        if provider == value:
            return alias
    return value


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

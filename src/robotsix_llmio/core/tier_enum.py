"""The legacy two-tier model selector :class:`Tier`.

Isolated in its own leaf module (no intra-package imports) so that both
:mod:`robotsix_llmio.core.provider` and
:mod:`robotsix_llmio.config.tier` can depend on it without forming an
import cycle. ``provider`` re-exports ``Tier`` for backwards-compatible
``from robotsix_llmio.core.provider import Tier`` call sites.
"""

from __future__ import annotations

from enum import StrEnum


class Tier(StrEnum):
    """**Deprecated** two-tier model selector — replaced by
    :class:`~robotsix_llmio.config.tier.TierConfig` and the integer *level*
    parameter on :meth:`LLMProvider.build_agent`.

    This enum remains for backward compatibility only.  Passing
    ``tier=Tier.CHEAP`` (or ``Tier.DEFAULT``) to ``build_agent()`` or
    ``new_model()`` still works but emits a :exc:`DeprecationWarning`.
    New code should use ``level=`` (1/2/3) and a ``TierConfig``.

    ========== =========== ============
    Member     Value       Replacement
    ========== =========== ============
    DEFAULT    ``default`` ``level=2``
    CHEAP      ``cheap``   ``level=1``
    ========== =========== ============
    """

    DEFAULT = "default"  # capable tier
    CHEAP = "cheap"  # fast/cheap tier

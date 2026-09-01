"""Two-slot, three-level configuration schema for provider+model bindings.

Prefer importing from :mod:`robotsix_llmio.core` — the public types are
re-exported there for discoverability.

Imports are deferred via :pep:`562` ``__getattr__`` so that importing the
config package does not eagerly pull in pydantic until a name is accessed.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

# Static re-declaration of every lazily-exported name (see ``__getattr__``
# below). These imports run ONLY under static analysis (``TYPE_CHECKING`` is
# False at runtime), so they add no import-time cost and preserve the PEP 562
# lazy-loading behaviour — but they let type checkers, IDEs, and CodeQL see
# each ``__all__`` entry as a defined module global. Without this, CodeQL's
# ``py/undefined-export`` query flags every ``__all__`` name as "exported but
# not defined" (it cannot model PEP 562 dynamic exports), failing the
# code-scanning check on any PR that adds a new export. Keep this block in
# sync with ``__all__`` and ``__getattr__``.
if TYPE_CHECKING:
    from robotsix_llmio.config.factory import create_model
    from robotsix_llmio.config.loader import TierConfigLoadError, load_tier_config
    from robotsix_llmio.config.tier import (
        DEFAULT_LEVEL1,
        DEFAULT_LEVEL2,
        DEFAULT_LEVEL3,
        FALLBACK_LEVEL1,
        FALLBACK_LEVEL2,
        FALLBACK_LEVEL3,
        VISION_DEFAULT,
        FailoverConfig,
        ProviderSlotConfig,
        TierConfig,
        TierLevel,
        TierLevelConfig,
    )
    from robotsix_llmio.core.factory import get_provider_for_identifier
    from robotsix_llmio.core.identifier import (
        MalformedIdentifierError,
        ParsedIdentifier,
        parse_model_identifier,
    )

__all__ = [
    "DEFAULT_LEVEL1",
    "DEFAULT_LEVEL2",
    "DEFAULT_LEVEL3",
    "FALLBACK_LEVEL1",
    "FALLBACK_LEVEL2",
    "FALLBACK_LEVEL3",
    "VISION_DEFAULT",
    "FailoverConfig",
    "MalformedIdentifierError",
    "ParsedIdentifier",
    "ProviderSlotConfig",
    "TierConfig",
    "TierConfigLoadError",
    "TierLevel",
    "TierLevelConfig",
    "create_model",
    "get_provider_for_identifier",
    "load_tier_config",
    "parse_model_identifier",
]

_TIER_NAMES = frozenset(
    {
        "DEFAULT_LEVEL1",
        "DEFAULT_LEVEL2",
        "DEFAULT_LEVEL3",
        "FALLBACK_LEVEL1",
        "FALLBACK_LEVEL2",
        "FALLBACK_LEVEL3",
        "FailoverConfig",
        "ProviderSlotConfig",
        "TierConfig",
        "TierLevel",
        "TierLevelConfig",
        "VISION_DEFAULT",
    }
)


def __getattr__(name: str) -> Any:  # PEP 562 — lazy imports
    if name in _TIER_NAMES:
        from . import tier

        return getattr(tier, name)
    if name in ("TierConfigLoadError", "load_tier_config"):
        from . import loader

        return getattr(loader, name)
    if name == "create_model":
        from . import factory

        return factory.create_model
    if name == "get_provider_for_identifier":
        from robotsix_llmio.core import factory as _core_factory

        return _core_factory.get_provider_for_identifier
    if name in (
        "MalformedIdentifierError",
        "ParsedIdentifier",
        "parse_model_identifier",
    ):
        from robotsix_llmio.core import identifier as _core_identifier

        return getattr(_core_identifier, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

"""Three-tier configuration schema for provider+model bindings.

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
        LEVEL1_DEFAULT,
        LEVEL2_DEFAULT,
        LEVEL3_DEFAULT,
        TierConfig,
        TierLevel,
        TierLevelConfig,
    )
    from robotsix_llmio.config.weekly_pace import ModelWeightConfig, WeeklyPaceConfig
    from robotsix_llmio.core.factory import get_provider_for_identifier
    from robotsix_llmio.core.identifier import (
        MalformedIdentifierError,
        ParsedIdentifier,
        parse_model_identifier,
    )

__all__ = [
    "LEVEL1_DEFAULT",
    "LEVEL2_DEFAULT",
    "LEVEL3_DEFAULT",
    "MalformedIdentifierError",
    "ModelWeightConfig",
    "ParsedIdentifier",
    "TierConfig",
    "TierConfigLoadError",
    "TierLevel",
    "TierLevelConfig",
    "WeeklyPaceConfig",
    "create_model",
    "get_provider_for_identifier",
    "load_tier_config",
    "parse_model_identifier",
]


def __getattr__(name: str) -> Any:  # PEP 562 — lazy imports
    if name == "LEVEL1_DEFAULT":
        from . import tier

        return tier.LEVEL1_DEFAULT
    if name == "LEVEL2_DEFAULT":
        from . import tier

        return tier.LEVEL2_DEFAULT
    if name == "LEVEL3_DEFAULT":
        from . import tier

        return tier.LEVEL3_DEFAULT
    if name == "TierConfig":
        from . import tier

        return tier.TierConfig
    if name == "TierLevel":
        from . import tier

        return tier.TierLevel
    if name == "TierLevelConfig":
        from . import tier

        return tier.TierLevelConfig
    if name == "TierConfigLoadError":
        from . import loader

        return loader.TierConfigLoadError
    if name == "load_tier_config":
        from . import loader

        return loader.load_tier_config
    if name == "ModelWeightConfig":
        from robotsix_llmio.weekly_pace import _config

        return _config.ModelWeightConfig
    if name == "WeeklyPaceConfig":
        from robotsix_llmio.weekly_pace import _config

        return _config.WeeklyPaceConfig
    if name == "create_model":
        from . import factory

        return factory.create_model
    if name == "get_provider_for_identifier":
        from robotsix_llmio.core import factory as _core_factory

        return _core_factory.get_provider_for_identifier
    if name == "MalformedIdentifierError":
        from robotsix_llmio.core import identifier as _core_identifier

        return _core_identifier.MalformedIdentifierError
    if name == "ParsedIdentifier":
        from robotsix_llmio.core import identifier as _core_identifier

        return _core_identifier.ParsedIdentifier
    if name == "parse_model_identifier":
        from robotsix_llmio.core import identifier as _core_identifier

        return _core_identifier.parse_model_identifier
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

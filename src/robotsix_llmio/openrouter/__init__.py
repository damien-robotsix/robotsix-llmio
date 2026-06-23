"""OpenRouter transport layer (model-family agnostic).

The model/provider (which import pydantic-ai and thus opentelemetry) are loaded
lazily via PEP 562 ``__getattr__`` so importing the lightweight ``transient``
helpers does not drag in pydantic-ai/OTel at module load.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from .transient import is_openrouter_transient, is_openrouter_upstream_error

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
    from ._async_client import AsyncOpenRouterClient
    from .model import OpenRouterModel, record_openrouter_cost
    from .provider import OpenRouterProvider
    from .provider_cost import (
        KeyUsage,
        OpenRouterKeyCostSource,
        OpenRouterProviderCostSource,
    )

__all__ = [
    "AsyncOpenRouterClient",
    "KeyUsage",
    "OpenRouterKeyCostSource",
    "OpenRouterModel",
    "OpenRouterProvider",
    "OpenRouterProviderCostSource",
    "is_openrouter_transient",
    "is_openrouter_upstream_error",
    "record_openrouter_cost",
]


def __getattr__(name: str) -> Any:  # PEP 562 — lazy heavy imports
    if name in ("OpenRouterModel", "record_openrouter_cost"):
        from . import model

        return getattr(model, name)
    if name == "OpenRouterProvider":
        from .provider import OpenRouterProvider

        return OpenRouterProvider
    if name in ("OpenRouterProviderCostSource", "OpenRouterKeyCostSource", "KeyUsage"):
        from . import provider_cost

        return getattr(provider_cost, name)
    if name == "AsyncOpenRouterClient":
        from . import _async_client

        return _async_client.AsyncOpenRouterClient
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

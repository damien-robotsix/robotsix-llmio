"""Deprecated shim — re-exports from ``robotsix_llmio.openrouter``.

The ``openrouter_deepseek`` module has been consolidated into
``robotsix_llmio.openrouter``.  This shim re-exports
``OpenRouterDeepseekModel`` and ``OpenRouterDeepseekProvider`` for
backward compatibility.

.. deprecated::
    Import from ``robotsix_llmio.openrouter`` instead:
    ``from robotsix_llmio.openrouter import OpenRouterDeepseekProvider``.
"""

from __future__ import annotations

import warnings
from typing import Any

__all__ = [
    "OpenRouterDeepseekModel",
    "OpenRouterDeepseekProvider",
]


def __getattr__(name: str) -> Any:  # PEP 562 — lazy heavy imports
    if name in ("OpenRouterDeepseekProvider", "OpenRouterDeepseekModel"):
        warnings.warn(
            f"Importing {name} from robotsix_llmio.openrouter_deepseek is "
            f"deprecated; import from robotsix_llmio.openrouter instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        from robotsix_llmio.openrouter._deepseek_model import OpenRouterDeepseekModel
        from robotsix_llmio.openrouter._deepseek_provider import (
            OpenRouterDeepseekProvider,
        )

        if name == "OpenRouterDeepseekProvider":
            return OpenRouterDeepseekProvider
        return OpenRouterDeepseekModel
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

"""Refdocs — direct HTTP access for documentation search and retrieval.

Replaces the agent-comm broker intermediary with direct REST calls.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from robotsix_llmio.exceptions import RobotsixLLMIOError


class RefdocsClientError(RobotsixLLMIOError):
    """Error from the refdocs client (HTTP, auth, malformed response)."""


# Static re-declaration of every lazily-exported name (see ``__getattr__``
# below). These imports run ONLY under static analysis (``TYPE_CHECKING`` is
# False at runtime), so they add no import-time cost and preserve the PEP 562
# lazy-loading behaviour — but they let type checkers, IDEs, and CodeQL see
# each ``__all__`` entry as a defined module global.
if TYPE_CHECKING:
    from ._async_client import AsyncRefdocsClient
    from ._settings import RefdocsSettings
    from .factory import build_refdocs_tools

__all__ = [
    "AsyncRefdocsClient",
    "RefdocsClientError",
    "RefdocsSettings",
    "build_refdocs_tools",
]


def __getattr__(name: str) -> Any:  # PEP 562 — lazy heavy imports
    if name == "AsyncRefdocsClient":
        from ._async_client import AsyncRefdocsClient

        return AsyncRefdocsClient
    if name == "RefdocsSettings":
        from ._settings import RefdocsSettings

        return RefdocsSettings
    if name == "build_refdocs_tools":
        from .factory import build_refdocs_tools

        return build_refdocs_tools
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

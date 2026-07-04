"""Direct-HTTP self-review client and pydantic-ai tool adapter.

Provides ``SelfReviewClient`` (async REST client for a self-review /
recent-activity API) and ``build_recent_activity_tools(client,
conversation_store)`` which wraps it into pydantic-ai-compatible async
tool functions.

Imports are deferred via :pep:`562` ``__getattr__`` so that importing the
self_review package does not eagerly pull in pydantic-ai until a name is
accessed.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from robotsix_llmio.exceptions import RobotsixLLMIOError


class SelfReviewClientError(RobotsixLLMIOError):
    """Error from the self-review client (HTTP, auth, malformed response)."""


# Static re-declaration of every lazily-exported name (see ``__getattr__``
# below). These imports run ONLY under static analysis (``TYPE_CHECKING`` is
# False at runtime), so they add no import-time cost and preserve the PEP 562
# lazy-loading behaviour — but they let type checkers, IDEs, and CodeQL see
# each ``__all__`` entry as a defined module global.
if TYPE_CHECKING:
    from ._client import SelfReviewClient, build_recent_activity_tools

__all__ = [
    "SelfReviewClient",
    "SelfReviewClientError",
    "build_recent_activity_tools",
]


def __getattr__(name: str) -> Any:  # PEP 562 — lazy heavy imports
    if name in ("SelfReviewClient", "build_recent_activity_tools"):
        from . import _client

        return getattr(_client, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

"""Direct-HTTP knowledge-store client and pydantic-ai tool adapter.

Provides ``KnowledgeClient`` (async REST client for a knowledge-store API)
and ``build_knowledge_tools(client)`` which wraps it into pydantic-ai-
compatible async tool functions.

Imports are deferred via :pep:`562` ``__getattr__`` so that importing the
knowledge package does not eagerly pull in pydantic-ai until a name is
accessed.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from robotsix_llmio.exceptions import RobotsixLLMIOError


class KnowledgeClientError(RobotsixLLMIOError):
    """Error from the knowledge-store client (HTTP, auth, malformed response)."""


# Static re-declaration of every lazily-exported name (see ``__getattr__``
# below). These imports run ONLY under static analysis (``TYPE_CHECKING`` is
# False at runtime), so they add no import-time cost and preserve the PEP 562
# lazy-loading behaviour — but they let type checkers, IDEs, and CodeQL see
# each ``__all__`` entry as a defined module global.
if TYPE_CHECKING:
    from ._client import KnowledgeClient, build_knowledge_tools

__all__ = [
    "KnowledgeClient",
    "KnowledgeClientError",
    "build_knowledge_tools",
]


def __getattr__(name: str) -> Any:  # PEP 562 — lazy heavy imports
    if name in ("KnowledgeClient", "build_knowledge_tools"):
        from . import _client

        return getattr(_client, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

"""Direct-HTTP REST clients — refdocs, knowledge, and self-review.

All three modules follow the same architectural pattern: an async
HTTP REST client plus a pydantic-ai tool-adapter wrapper, replacing
the agent-comm broker intermediary with direct REST calls.

Shared infrastructure lives in ``_base.py`` (``BaseHttpClient``).
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .knowledge import KnowledgeClient, KnowledgeClientError, build_knowledge_tools
    from .refdocs import (
        AsyncRefdocsClient,
        RefdocsClientError,
        RefdocsSettings,
        build_refdocs_tools,
    )
    from .self_review import (
        SelfReviewClient,
        SelfReviewClientError,
        build_recent_activity_tools,
    )

__all__ = [
    # refdocs
    "AsyncRefdocsClient",
    "RefdocsClientError",
    "RefdocsSettings",
    "build_refdocs_tools",
    # knowledge
    "KnowledgeClient",
    "KnowledgeClientError",
    "build_knowledge_tools",
    # self_review
    "SelfReviewClient",
    "SelfReviewClientError",
    "build_recent_activity_tools",
]


def __getattr__(name: str) -> Any:  # PEP 562 — lazy heavy imports
    if name in ("AsyncRefdocsClient", "RefdocsClientError"):
        from .refdocs import AsyncRefdocsClient, RefdocsClientError

        return {"AsyncRefdocsClient": AsyncRefdocsClient, "RefdocsClientError": RefdocsClientError}[name]
    if name in ("RefdocsSettings", "build_refdocs_tools"):
        from .refdocs import RefdocsSettings, build_refdocs_tools

        return {"RefdocsSettings": RefdocsSettings, "build_refdocs_tools": build_refdocs_tools}[name]
    if name in ("KnowledgeClient", "KnowledgeClientError", "build_knowledge_tools"):
        from . import knowledge

        return getattr(knowledge, name)
    if name in ("SelfReviewClient", "SelfReviewClientError", "build_recent_activity_tools"):
        from . import self_review

        return getattr(self_review, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

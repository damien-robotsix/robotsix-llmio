"""robotsix-llmio: provider-agnostic LLM I/O with derived per-provider layers."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from .exceptions import RobotsixLLMIOError

# Static re-declaration of the lazily-exported names (see ``__getattr__``
# below). These imports run ONLY under static analysis (``TYPE_CHECKING`` is
# False at runtime), so they add no import-time cost and preserve the lazy
# loading — but they let type checkers, IDEs, and CodeQL see each ``__all__``
# entry as a defined module global (otherwise CodeQL's ``py/undefined-export``
# query flags every ``__all__`` name as "exported but not defined"). Keep this
# block in sync with ``__all__`` and ``__getattr__``.
if TYPE_CHECKING:
    from .clients.knowledge import (
        KnowledgeClient,
        KnowledgeClientError,
        build_knowledge_tools,
    )
    from .clients.refdocs import (
        AsyncRefdocsClient,
        RefdocsClientError,
        build_refdocs_tools,
    )
    from .clients.self_review import (
        SelfReviewClient,
        SelfReviewClientError,
        build_recent_activity_tools,
    )
    from .core.factory import (
        build_agent_for_level,
        default_tier_config,
        get_provider_for_level,
    )
    from .core.langfuse_client import LangfuseClientError
    from .openrouter import OpenRouterAPIError

__all__ = [
    "AsyncRefdocsClient",
    "KnowledgeClient",
    "KnowledgeClientError",
    "LangfuseClientError",
    "OpenRouterAPIError",
    "RefdocsClientError",
    "RobotsixLLMIOError",
    "SelfReviewClient",
    "SelfReviewClientError",
    "build_agent_for_level",
    "build_knowledge_tools",
    "build_recent_activity_tools",
    "build_refdocs_tools",
    "default_tier_config",
    "get_provider_for_level",
]


def __getattr__(name: str) -> Any:  # PEP 562 — lazy heavy imports
    if name == "build_agent_for_level":
        from .core import factory

        return factory.build_agent_for_level
    if name == "get_provider_for_level":
        from .core import factory

        return factory.get_provider_for_level
    if name == "default_tier_config":
        from .core import factory

        return factory.default_tier_config
    if name in ("KnowledgeClient", "build_knowledge_tools", "KnowledgeClientError"):
        from .clients import knowledge

        return getattr(knowledge, name)
    if name == "LangfuseClientError":
        from .core import langfuse_client

        return langfuse_client.LangfuseClientError
    if name == "OpenRouterAPIError":
        from . import openrouter

        return openrouter.OpenRouterAPIError
    if name in (
        "AsyncRefdocsClient",
        "build_refdocs_tools",
        "RefdocsClientError",
    ):
        from .clients import refdocs

        return getattr(refdocs, name)
    if name in (
        "SelfReviewClient",
        "build_recent_activity_tools",
        "SelfReviewClientError",
    ):
        from .clients import self_review

        return getattr(self_review, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

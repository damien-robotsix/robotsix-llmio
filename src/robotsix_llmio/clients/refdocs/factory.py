"""Tool factory — builds pydantic-ai Tools for refdocs access.

The returned tools make direct HTTP calls to the refdocs REST API via
:class:`AsyncRefdocsClient` — no agent-comm broker intermediary.
"""

from __future__ import annotations

from typing import Any

from ._async_client import AsyncRefdocsClient
from ._settings import RefdocsSettings


def build_refdocs_tools(settings: RefdocsSettings) -> list[Any]:
    """Build pydantic-ai :class:`~pydantic_ai.tools.Tool` objects wrapping
    direct-HTTP refdocs access.

    Parameters
    ----------
    settings:
        Configuration for the refdocs HTTP client (base URL, auth, timeout).

    Returns
    -------
    list of pydantic_ai.Tool
        Tools ready to pass to a pydantic-ai ``Agent`` via ``tools=``.
    """
    import pydantic_ai

    client = AsyncRefdocsClient(
        base_url=settings.base_url,
        api_key=settings.resolved_api_key,
        request_timeout=settings.request_timeout,
    )

    async def _search_refdocs(query: str) -> str:
        """Search the project documentation for *query*.

        Returns a JSON-formatted list of matching documentation entries
        with their paths and titles.
        """
        import json

        results = await client.search(query)
        return json.dumps(results, ensure_ascii=False, indent=2)

    async def _get_refdocs(path: str) -> str:
        """Retrieve the full content of the documentation page at *path*.

        Returns the document body as plain text.
        """
        return await client.get_doc(path)

    return [
        pydantic_ai.Tool(
            _search_refdocs,
            name="search_refdocs",
            description="Search the project documentation. Returns matching "
            "entries with path and title.",
        ),
        pydantic_ai.Tool(
            _get_refdocs,
            name="get_refdocs",
            description="Retrieve the full content of a documentation page by "
            "its path (e.g. 'core/index').",
        ),
    ]

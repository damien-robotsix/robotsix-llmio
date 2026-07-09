"""Async refdocs REST client for searching and retrieving documentation.

Self-contained: depends only on ``httpx`` and the public REST API — no
pydantic-ai, no OTel. Direct HTTP access replaces the agent-comm broker.
"""

from __future__ import annotations

from typing import Any

from robotsix_llmio.clients._base import BaseHttpClient
from robotsix_llmio.clients.refdocs import RefdocsClientError

from ._settings import _DEFAULT_BASE_URL


class AsyncRefdocsClient(BaseHttpClient):
    """Async refdocs REST client for searching and retrieving documentation.

    Hits the refdocs REST API directly (no agent-comm broker). Creates a
    fresh, timeout-bounded ``httpx.AsyncClient`` per request (stateless
    per-call pattern).
    """

    # ------------------------------------------------------------------ #
    #  BaseHttpClient contract
    # ------------------------------------------------------------------ #

    @property
    def _error_type(self) -> type[Exception]:
        return RefdocsClientError

    @property
    def _error_label(self) -> str:
        return "Refdocs"

    # ------------------------------------------------------------------ #
    #  Constructor
    # ------------------------------------------------------------------ #

    def __init__(
        self,
        *,
        base_url: str | None = None,
        api_key: str | None = None,
        request_timeout: float = 30.0,
    ) -> None:
        super().__init__(
            base_url=base_url or _DEFAULT_BASE_URL,
            api_key=api_key,
            request_timeout=request_timeout,
        )

    # ------------------------------------------------------------------ #
    #  Public API
    # ------------------------------------------------------------------ #

    async def search(self, query: str) -> list[dict[str, Any]]:
        """Search the documentation index for *query*.

        Hits ``GET {base_url}/search?q={query}``. Returns a list of result
        dicts, each with at least ``"path"`` and ``"title"`` keys.

        Raises ``RefdocsClientError`` on any non-2xx response or
        network failure.
        """
        data = await self._get("/search", params={"q": query})
        results = data.get("results")
        if isinstance(results, list):
            return results
        return []

    async def get_doc(self, path: str) -> str:
        """Retrieve the full content of the documentation at *path*.

        Hits ``GET {base_url}/docs/{path}``. Returns the document body as
        a string.

        Raises ``RefdocsClientError`` on any non-2xx response or
        network failure.
        """
        data = await self._get(f"/docs/{path}")
        return str(data.get("content") or "")

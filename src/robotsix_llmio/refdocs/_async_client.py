"""Async refdocs REST client for searching and retrieving documentation.

Self-contained: depends only on ``httpx`` and the public REST API — no
pydantic-ai, no OTel. Direct HTTP access replaces the agent-comm broker.
"""

from __future__ import annotations

from typing import Any, cast

from ..core.http import timeout_http_client
from ._base import _DEFAULT_BASE_URL


class AsyncRefdocsClient:
    """Async refdocs REST client for searching and retrieving documentation.

    Hits the refdocs REST API directly (no agent-comm broker). Creates a
    fresh, timeout-bounded ``httpx.AsyncClient`` per request (stateless
    per-call pattern).
    """

    def __init__(
        self,
        *,
        base_url: str | None = None,
        api_key: str | None = None,
        request_timeout: float = 30.0,
    ) -> None:
        self._base_url = (base_url or _DEFAULT_BASE_URL).rstrip("/")
        self._api_key = api_key
        self._request_timeout = request_timeout

    # ------------------------------------------------------------------ #
    #  Public API
    # ------------------------------------------------------------------ #

    async def search(self, query: str) -> list[dict[str, Any]]:
        """Search the documentation index for *query*.

        Hits ``GET {base_url}/search?q={query}``. Returns a list of result
        dicts, each with at least ``"path"`` and ``"title"`` keys.

        Raises ``RuntimeError`` on any non-2xx response.
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

        Raises ``RuntimeError`` on any non-2xx response.
        """
        data = await self._get(f"/docs/{path}")
        return str(data.get("content") or "")

    # ------------------------------------------------------------------ #
    #  Internal
    # ------------------------------------------------------------------ #

    async def _get(
        self, path: str, *, params: dict[str, str] | None = None
    ) -> dict[str, Any]:
        """Send a GET to *path* and return the JSON response body.

        Raises ``RuntimeError`` on any non-2xx response.
        """
        import httpx

        url = f"{self._base_url}{path}"
        headers: dict[str, str] = {}
        if self._api_key:
            headers["Authorization"] = f"Bearer {self._api_key}"
        async with timeout_http_client() as client:
            client.timeout = httpx.Timeout(self._request_timeout)
            resp = await client.get(url, headers=headers, params=params)
        if not (200 <= resp.status_code < 300):
            raise RuntimeError(
                f"Refdocs {path} request failed: HTTP {resp.status_code}"
            )
        return cast(dict[str, Any], resp.json())

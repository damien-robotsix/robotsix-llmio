"""Async HTTP client for a knowledge-store REST API and pydantic-ai tool adapter.

``KnowledgeClient`` makes direct HTTP calls to a knowledge-store REST API
(no agent-comm broker). ``build_knowledge_tools(client)`` wraps it into
pydantic-ai-compatible async tool functions so an agent can search and
retrieve documents.
"""

from __future__ import annotations

from typing import Any

from robotsix_llmio.core.http import timeout_http_client
from robotsix_llmio.knowledge import KnowledgeClientError

_DEFAULT_BASE_URL = "http://localhost:8000/api/v1"


class KnowledgeClient:
    """Async HTTP client for a knowledge-store REST API.

    Connects directly to the knowledge-store HTTP API — no agent-comm
    broker intermediary. Creates a fresh, timeout-bounded
    ``httpx.AsyncClient`` per request (stateless per-call pattern).

    Parameters
    ----------
    base_url:
        Root URL of the knowledge-store API (e.g. ``http://knowledge:8000/api/v1``).
    api_key:
        Optional bearer token sent as ``Authorization: Bearer <api_key>``.
    """

    def __init__(
        self,
        *,
        base_url: str = _DEFAULT_BASE_URL,
        api_key: str | None = None,
    ) -> None:
        self._base_url = base_url.rstrip("/")
        self._api_key = api_key

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #

    async def search(self, query: str, top_k: int = 10) -> list[dict[str, Any]]:
        """Full-text search the knowledge store.

        Parameters
        ----------
        query:
            Free-text search query.
        top_k:
            Maximum number of results to return (default 10).

        Returns
        -------
        list[dict]
            A list of result dicts, each with keys ``id``, ``title``,
            ``snippet``, and ``score``. An empty list means no matches.
        """
        params: dict[str, str | int] = {"q": query, "top_k": top_k}
        data = await self._get("/search", params=params)
        results: list[dict[str, Any]] = data.get("results", [])
        return results

    async def get_document(self, doc_id: str) -> dict[str, Any]:
        """Retrieve a single document by its identifier.

        Parameters
        ----------
        doc_id:
            The document identifier.

        Returns
        -------
        dict
            The document with keys ``id``, ``title``, ``content``, and
            ``metadata``.

        Raises
        ------
        KnowledgeClientError
            If the document is not found (HTTP 404) or the request fails.
        """
        return await self._get(f"/documents/{doc_id}")

    # ------------------------------------------------------------------ #
    # Internal helpers
    # ------------------------------------------------------------------ #

    async def _get(
        self,
        path: str,
        params: dict[str, str | int] | None = None,
    ) -> dict[str, Any]:
        """Send a GET to *path* and return the JSON response body.

        Raises ``KnowledgeClientError`` on any non-2xx response or
        network failure.
        """
        url = f"{self._base_url}{path}"
        headers: dict[str, str] = {}
        if self._api_key is not None:
            headers["Authorization"] = f"Bearer {self._api_key}"

        try:
            async with timeout_http_client() as client:
                resp = await client.get(url, headers=headers, params=params)
        except Exception as exc:
            raise KnowledgeClientError(
                f"Knowledge store request to {path} failed: {exc}"
            ) from exc

        if not (200 <= resp.status_code < 300):
            raise KnowledgeClientError(
                f"Knowledge store {path} returned HTTP {resp.status_code}"
            )
        try:
            body = resp.json()
        except Exception as exc:
            raise KnowledgeClientError(
                f"Knowledge store {path} returned a non-JSON body"
            ) from exc
        if not isinstance(body, dict):
            raise KnowledgeClientError(
                f"Knowledge store {path} returned unexpected JSON shape"
                " (expected object)"
            )
        return body


# ---------------------------------------------------------------------- #
# pydantic-ai tool adapter
# ---------------------------------------------------------------------- #


def build_knowledge_tools(client: KnowledgeClient) -> list[Any]:
    """Build pydantic-ai-compatible async tool functions from a
    :class:`KnowledgeClient`.

    Returns two async tool functions:

    * ``search_knowledge(query: str) -> str`` — full-text search, returns
      formatted results.
    * ``get_knowledge_document(doc_id: str) -> str`` — retrieve a document
      by ID, returns the document content.

    Each tool handles its own errors and returns a plain-text message so
    the agent can interpret the result directly.
    """

    async def search_knowledge(query: str) -> str:
        """Full-text search the knowledge store.

        Use this to find documents relevant to a topic or question.
        Returns formatted search results with document IDs, titles,
        and relevance scores.
        """
        try:
            results = await client.search(query)
        except KnowledgeClientError as exc:
            return f"Search failed: {exc}"

        if not results:
            return f"No documents found for query: {query}"

        lines: list[str] = []
        for r in results:
            doc_id = r.get("id", "?")
            title = r.get("title", "Untitled")
            snippet = r.get("snippet", "")
            score = r.get("score", 0.0)
            lines.append(f"- [{doc_id}] {title} (score={score:.2f})\n  {snippet}")
        return "\n".join(lines)

    async def get_knowledge_document(doc_id: str) -> str:
        """Retrieve a specific document from the knowledge store by ID.

        Use this after ``search_knowledge`` to read the full content of
        a relevant document. Returns the document title and content.
        """
        try:
            doc = await client.get_document(doc_id)
        except KnowledgeClientError as exc:
            return f"Failed to retrieve document {doc_id}: {exc}"

        title = doc.get("title", "Untitled")
        content = doc.get("content", "")
        if not content:
            return f"Document {doc_id} ({title}) has no content."
        return f"# {title}\n\n{content}"

    return [search_knowledge, get_knowledge_document]

"""Async Langfuse REST read client.

Mirrors :class:`LangfuseReadClient` with ``httpx.AsyncClient`` — the
async kernel for consumers that run inside an event loop (e.g.
FastAPI applications).
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from datetime import UTC, datetime, timedelta
from typing import Any
from urllib.parse import urlparse

import httpx

from .constants import HTTP_CLIENT_TIMEOUT
from .langfuse_client import (
    _TRACES_PATH,
    LangfuseClientError,
    _LangfuseReadClientBase,
    _observation_cost,
    _observation_provider,
    _parse_timestamp,
)

__all__ = ["AsyncLangfuseReadClient"]


class AsyncLangfuseReadClient(_LangfuseReadClientBase):
    """Async variant of :class:`LangfuseReadClient`.

    Owns the same reusable kernel — auth header, base-URL resolution,
    and pagination — but uses ``httpx.AsyncClient`` so it is safe to
    call from within an event loop.

    Credentials are always passed in explicitly; the client reads no
    ``LANGFUSE_*`` env vars.
    """

    async def aiter_pages(
        self,
        url: str,
        params: dict[str, Any] | None = None,
        *,
        error_label: str = "request",
    ) -> AsyncIterator[list[dict[str, Any]]]:
        """Paginate *url* (1-based ``page``), yielding each page's ``data``.

        Owns the ``httpx.AsyncClient``, the page loop, the ``Basic`` auth
        header, the non-2xx ``LangfuseClientError`` (labelled with *error_label*),
        the empty-``data`` break, and the ``meta.totalPages`` termination.
        *url* may be an absolute URL or a path relative to :attr:`base_url`.
        """
        target = url if urlparse(url).scheme else self.url(url)
        base_params = dict(params or {})
        headers = {"Authorization": self.auth_header()}
        async with httpx.AsyncClient(timeout=HTTP_CLIENT_TIMEOUT) as client:
            page = 1
            while True:
                resp = await client.get(
                    target,
                    params={**base_params, "page": page},
                    headers=headers,
                )
                if not (200 <= resp.status_code < 300):
                    raise LangfuseClientError(
                        f"Langfuse {error_label} failed: HTTP {resp.status_code}"
                    )
                body = resp.json()
                data = body.get("data") or []
                if not data:
                    break
                yield data
                meta = body.get("meta")
                if isinstance(meta, dict):
                    total_pages = meta.get("totalPages")
                    if isinstance(total_pages, int) and page >= total_pages:
                        break
                page += 1

    async def fetch_traces_window(
        self,
        hours: float,
    ) -> AsyncIterator[dict[str, Any]]:
        """Yield individual traces created in the last *hours* hours.

        Convenience wrapper that computes ``fromTimestamp`` and flattens
        the paginated pages into a stream of individual trace dicts.
        """
        from_timestamp = (datetime.now(UTC) - timedelta(hours=hours)).isoformat()
        params: dict[str, Any] = {"fromTimestamp": from_timestamp}
        async for page in self.aiter_pages(
            _TRACES_PATH, params, error_label="traces window"
        ):
            for trace in page:
                yield trace

    async def fetch_trace_detail(self, trace_id: str) -> dict[str, Any]:
        """Fetch a single trace by its Langfuse trace ID."""
        target = self.url(f"{_TRACES_PATH}/{trace_id}")
        headers = {"Authorization": self.auth_header()}
        async with httpx.AsyncClient(timeout=HTTP_CLIENT_TIMEOUT) as client:
            resp = await client.get(target, headers=headers)
            if not (200 <= resp.status_code < 300):
                raise LangfuseClientError(
                    f"Langfuse trace detail failed: HTTP {resp.status_code}"
                )
            result: dict[str, Any] = resp.json()
            return result

    @staticmethod
    def parse_timestamp(value: Any) -> datetime:
        """Parse an ISO-8601 timestamp (tolerating a trailing ``Z``)."""
        return _parse_timestamp(value)

    @staticmethod
    def observation_provider(obs: dict[str, Any]) -> str | None:
        """Pull the ``provider`` tag out of an observation's metadata."""
        return _observation_provider(obs)

    @staticmethod
    def observation_cost(obs: dict[str, Any]) -> float:
        """USD cost of one Langfuse observation."""
        return _observation_cost(obs)

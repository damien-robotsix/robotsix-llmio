"""Async OpenRouter REST client for per-key usage and account credits.

Self-contained: depends only on ``httpx`` and the public REST API — no
pydantic-ai, no OTel. Provides the async counterpart to the sync
``OpenRouterKeyCostSource`` plus account-level credits.
"""

from __future__ import annotations

from typing import Any

from ..core.http import timeout_http_client
from ._base import _DEFAULT_BASE_URL
from .provider_cost import KeyUsage


class AsyncOpenRouterClient:
    """Async OpenRouter REST client for per-key usage and account credits.

    Hits the OpenRouter public REST API with Bearer-auth. Creates a fresh,
    timeout-bounded ``httpx.AsyncClient`` per request (stateless per-call
    pattern, matches the sync ``OpenRouterKeyCostSource``).
    """

    def __init__(self, *, api_key: str, base_url: str | None = None) -> None:
        self._key = api_key
        self._base_url = (base_url or _DEFAULT_BASE_URL).rstrip("/")

    async def fetch_key_usage(self) -> KeyUsage:
        """Current cumulative usage of the authenticating key.

        Hits ``GET {base_url}/auth/key``.

        Raises ``RuntimeError`` on any non-2xx response.
        """
        data = await self._get("/auth/key")
        limit = data.get("limit")
        return KeyUsage(
            usage=float(data.get("usage", 0) or 0),
            limit=None if limit is None else float(limit),
            label=data.get("label"),
        )

    async def fetch_credits(self) -> dict[str, float]:
        """Account-level credit balance.

        Hits ``GET {base_url}/credits``. Returns a dict with keys
        ``total_credits``, ``total_usage``, and ``remaining`` (each
        rounded to 6 decimal places).

        Raises ``RuntimeError`` on any non-2xx response.
        """
        data = await self._get("/credits")
        total_credits = float(data.get("total_credits") or 0.0)
        total_usage = float(data.get("total_usage") or 0.0)
        remaining = total_credits - total_usage
        return {
            "total_credits": round(total_credits, 6),
            "total_usage": round(total_usage, 6),
            "remaining": round(remaining, 6),
        }

    async def _get(self, path: str) -> dict[str, Any]:
        """Send a GET to *path* and return the ``"data"`` key of the JSON
        response body.

        Raises ``RuntimeError`` on any non-2xx response.
        """
        url = f"{self._base_url}{path}"
        headers = {"Authorization": f"Bearer {self._key}"}
        async with timeout_http_client() as client:
            resp = await client.get(url, headers=headers)
        if not (200 <= resp.status_code < 300):
            raise RuntimeError(
                f"OpenRouter {path} request failed: "
                f"HTTP {resp.status_code}: {resp.text[:200]}"
            )
        return resp.json().get("data") or {}

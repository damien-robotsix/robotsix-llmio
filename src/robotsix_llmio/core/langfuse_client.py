"""First-class client for the Langfuse REST API on the read path.

The reusable Langfuse-protocol kernel: ``Basic`` auth-header construction,
``https://cloud.langfuse.com`` base-URL resolution, paginated ``GET`` over the
public traces and observations endpoints, and the trace/observation parsing
helpers (``totalCost`` / ISO-8601 timestamps).

This is the single module in the fleet that speaks the Langfuse REST read
protocol. Higher-level adapters compose :class:`LangfuseReadClient` rather than
re-deriving auth/base-url/pagination/parsing:

* :class:`robotsix_llmio.core.langfuse_cost.LangfuseCostLogSource` builds the
  window-aggregate :class:`~robotsix_llmio.core.cost_log.CostLogSource` on top
  of it.
* Downstream consumers (e.g. ``robotsix_mill``) layer their own
  session/observation/aggregation helpers on the same kernel.

Self-contained: depends only on ``httpx`` and the public Langfuse REST API — no
Langfuse SDK, no ``tracing`` extra. It bakes no credentials: the consumer always
constructs it with explicit keys.
"""

from __future__ import annotations

import base64
from collections.abc import Iterator
from datetime import datetime
from typing import Any

import httpx

from ._otel import _DEFAULT_LANGFUSE_BASE_URL as _DEFAULT_BASE_URL
from ._otel import LANGFUSE_COST_DETAILS_TOTAL_KEY
from .constants import HTTP_CLIENT_TIMEOUT

_PAGE_LIMIT = 100
_TRACES_PATH = "/api/public/traces"
_OBSERVATIONS_PATH = "/api/public/observations"


class LangfuseReadClient:
    """Speak the Langfuse REST read protocol.

    Owns the reusable kernel — auth header, base-URL resolution, and 1-based
    pagination over the public endpoints — so adapters need not re-derive it.
    Credentials are always passed in explicitly; the client reads no
    ``LANGFUSE_*`` env vars (env defaulting, if any, belongs to the consumer).
    """

    def __init__(
        self,
        *,
        public_key: str,
        secret_key: str,
        base_url: str | None = None,
    ) -> None:
        self._public_key = public_key
        self._secret_key = secret_key
        self._base_url = (base_url or _DEFAULT_BASE_URL).rstrip("/")

    @property
    def base_url(self) -> str:
        """The resolved base URL (default ``https://cloud.langfuse.com``)."""
        return self._base_url

    def url(self, path: str) -> str:
        """Join *path* (e.g. :data:`_TRACES_PATH`) onto the base URL."""
        return f"{self._base_url}/{path.lstrip('/')}"

    def auth_header(self) -> str:
        """Build the ``Basic <base64(public:secret)>`` Authorization header."""
        token = base64.b64encode(
            f"{self._public_key}:{self._secret_key}".encode()
        ).decode()
        return f"Basic {token}"

    def iter_pages(
        self,
        url: str,
        params: dict[str, Any] | None = None,
        *,
        error_label: str = "request",
    ) -> Iterator[list[dict[str, Any]]]:
        """Paginate *url* (1-based ``page``), yielding each page's ``data``.

        Owns the ``httpx.Client``, the page loop, the ``Basic`` auth header, the
        non-2xx ``RuntimeError`` (labelled with *error_label*), the
        empty-``data`` break, and the ``meta.totalPages`` termination. *url* may
        be an absolute URL or a path relative to :attr:`base_url`.
        """
        target = url if "://" in url else self.url(url)
        base_params = dict(params or {})
        headers = {"Authorization": self.auth_header()}
        with httpx.Client(timeout=HTTP_CLIENT_TIMEOUT) as client:
            page = 1
            while True:
                resp = client.get(
                    target,
                    params={**base_params, "page": page},
                    headers=headers,
                )
                if not (200 <= resp.status_code < 300):
                    raise RuntimeError(
                        f"Langfuse {error_label} failed: "
                        f"HTTP {resp.status_code}: {resp.text[:200]}"
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


def _observation_provider(obs: dict[str, Any]) -> str | None:
    """Pull the ``provider`` tag out of a Langfuse observation's metadata.

    The write path stamps ``langfuse.observation.metadata.provider``, which
    Langfuse surfaces under the observation's ``metadata`` dict.
    """
    metadata = obs.get("metadata")
    if isinstance(metadata, dict):
        provider = metadata.get("provider")
        if provider is not None:
            return str(provider)
    return None


def _observation_cost(obs: dict[str, Any]) -> float:
    """USD cost of one Langfuse observation.

    Prefers Langfuse's server-computed ``calculatedTotalCost``, then a raw
    ``totalCost``, then ``costDetails.total`` (mirrors the write-side rollup).
    """
    for key in ("calculatedTotalCost", "totalCost"):
        value = obs.get(key)
        if value is not None:
            return float(value or 0)
    cost_details = obs.get("costDetails")
    if (
        isinstance(cost_details, dict)
        and cost_details.get(LANGFUSE_COST_DETAILS_TOTAL_KEY) is not None
    ):
        return float(cost_details[LANGFUSE_COST_DETAILS_TOTAL_KEY] or 0)
    return 0.0


def _parse_timestamp(value: Any) -> datetime:
    """Parse an ISO-8601 timestamp (tolerating a trailing ``Z``)."""
    if isinstance(value, datetime):
        return value
    text = str(value or "")
    if text.endswith("Z"):
        text = f"{text[:-1]}+00:00"
    return datetime.fromisoformat(text)

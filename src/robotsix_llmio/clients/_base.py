"""Shared base class for direct-HTTP REST clients.

Provides ``BaseHttpClient`` with a common ``_get()`` implementation
shared by refdocs, knowledge, and self_review clients.
"""

from __future__ import annotations

from typing import Any

from robotsix_llmio.core.http import timeout_http_client


class BaseHttpClient:
    """Shared base for direct-HTTP REST clients.

    Provides a common ``_get()`` implementation that handles URL
    construction, auth headers, timeout-bounded HTTP GET, status
    checking, JSON parsing, and shape validation.  Subclasses only
    need to supply the error type and error-message label via the
    ``_error_type`` and ``_error_label`` properties.

    Parameters
    ----------
    base_url:
        Root URL of the REST API, e.g. ``http://service:8000/api/v1``.
    api_key:
        Optional bearer token sent as ``Authorization: Bearer <api_key>``.
    request_timeout:
        Optional per-request timeout in seconds. When ``None``, the
        default timeout of the underlying ``httpx.AsyncClient`` is used.
    """

    def __init__(
        self,
        *,
        base_url: str,
        api_key: str | None = None,
        request_timeout: float | None = None,
    ) -> None:
        self._base_url = base_url.rstrip("/")
        self._api_key = api_key
        self._request_timeout = request_timeout

    # ------------------------------------------------------------------ #
    # Subclass contract
    # ------------------------------------------------------------------ #

    @property
    def _error_type(self) -> type[Exception]:
        """Error class that ``_get()`` raises on failure.

        Subclasses MUST override this property — e.g.
        ``return KnowledgeClientError``.
        """
        raise NotImplementedError

    @property
    def _error_label(self) -> str:
        """Human-readable label used in error messages.

        Subclasses MUST override this property — e.g.
        ``return "Knowledge store"``.
        """
        raise NotImplementedError

    # ------------------------------------------------------------------ #
    # Shared HTTP helper
    # ------------------------------------------------------------------ #

    async def _get(
        self,
        path: str,
        params: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Send a GET to *path* and return the JSON response body.

        Raises the subclass's ``_error_type`` on any non-2xx response,
        network failure, non-JSON body, or unexpected JSON shape.
        """
        url = f"{self._base_url}{path}"
        headers: dict[str, str] = {}
        if self._api_key is not None:
            headers["Authorization"] = f"Bearer {self._api_key}"

        try:
            async with timeout_http_client() as client:
                if self._request_timeout is not None:
                    import httpx

                    client.timeout = httpx.Timeout(self._request_timeout)
                resp = await client.get(url, headers=headers, params=params)
        except Exception as exc:
            raise self._error_type(
                f"{self._error_label} request to {path} failed: {exc}"
            ) from exc

        if not (200 <= resp.status_code < 300):
            raise self._error_type(
                f"{self._error_label} {path} returned HTTP {resp.status_code}"
            )
        try:
            body = resp.json()
        except Exception as exc:
            raise self._error_type(
                f"{self._error_label} {path} returned a non-JSON body"
            ) from exc
        if not isinstance(body, dict):
            raise self._error_type(
                f"{self._error_label} {path} returned unexpected JSON shape"
                " (expected object)"
            )
        return body

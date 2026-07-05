"""Shared REST client helpers — GET + JSON response parsing.

Internal module. Provides a reusable :func:`_get_json` async helper that
knowledge, self-review, and refdocs clients all use, plus the shared
``_DEFAULT_BASE_URL`` constant for services running on the standard
``8000/api/v1`` layout.
"""

from __future__ import annotations

from typing import Any

from .http import timeout_http_client

_DEFAULT_BASE_URL = "http://localhost:8000/api/v1"


async def _get_json(
    *,
    base_url: str,
    path: str,
    params: dict[str, str | int] | None = None,
    api_key: str | None = None,
    timeout_seconds: float | None = None,
    error_cls: type[Exception],
    error_label: str,
) -> dict[str, Any]:
    """Send an authenticated GET request and return the parsed JSON body.

    Parameters
    ----------
    base_url:
        Root URL of the REST API (without trailing slash).
    path:
        URL path (e.g. ``"/search"``).  Appended directly to *base_url*.
    params:
        Optional query-string parameters.
    api_key:
        Optional bearer token, sent as ``Authorization: Bearer <api_key>``.
    timeout_seconds:
        Per-request timeout in seconds.  When ``None`` the default from
        :func:`~robotsix_llmio.core.http.timeout_http_client` is used.
    error_cls:
        Exception class to raise on failure (must accept a single
        ``str`` argument).
    error_label:
        Human-readable service label for error messages (e.g.
        ``"Knowledge store"``, ``"Self-review"``, ``"Refdocs"``).

    Returns
    -------
    dict[str, Any]
        The parsed JSON response body.

    Raises
    ------
    error_cls
        On any network failure, non-2xx status, non-JSON body, or
        non-dict JSON shape.
    """
    import httpx

    url = f"{base_url}{path}"
    headers: dict[str, str] = {}
    if api_key is not None:
        headers["Authorization"] = f"Bearer {api_key}"

    try:
        async with timeout_http_client() as client:
            if timeout_seconds is not None:
                client.timeout = httpx.Timeout(timeout_seconds)
            resp = await client.get(url, headers=headers, params=params)
    except Exception as exc:
        raise error_cls(
            f"{error_label} request to {path} failed: {exc}"
        ) from exc

    if not (200 <= resp.status_code < 300):
        raise error_cls(
            f"{error_label} {path} returned HTTP {resp.status_code}"
        )
    try:
        body = resp.json()
    except Exception as exc:
        raise error_cls(
            f"{error_label} {path} returned a non-JSON body"
        ) from exc
    if not isinstance(body, dict):
        raise error_cls(
            f"{error_label} {path} returned unexpected JSON shape"
            " (expected object)"
        )
    return body

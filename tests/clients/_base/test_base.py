"""Unit tests for ``BaseHttpClient`` — the shared base class for
direct-HTTP REST clients.

Drives ``BaseHttpClient`` via ``httpx.MockTransport`` (no network,
no respx dependency).  Async methods are called via ``asyncio.run`` —
no ``pytest-asyncio`` needed.
"""

from __future__ import annotations

import asyncio

import httpx
import pytest

from robotsix_llmio.clients import _base as _base_module
from robotsix_llmio.clients._base import BaseHttpClient
from tests.core.conftest import install_timeout_transport

# --------------------------------------------------------------------------- #
# Test-only concrete subclass
# --------------------------------------------------------------------------- #


class _TestClientError(Exception):
    """Custom error raised by the test concrete subclass."""


class _TestClient(BaseHttpClient):
    """Minimal concrete ``BaseHttpClient`` for unit testing."""

    @property
    def _error_type(self) -> type[Exception]:
        return _TestClientError

    @property
    def _error_label(self) -> str:
        return "Test client"


# --------------------------------------------------------------------------- #
# __init__
# --------------------------------------------------------------------------- #


def test_init_strips_trailing_slash_from_base_url() -> None:
    client = _TestClient(base_url="http://example.com/api/")
    assert client._base_url == "http://example.com/api"


def test_init_preserves_url_without_trailing_slash() -> None:
    client = _TestClient(base_url="http://example.com/api")
    assert client._base_url == "http://example.com/api"


def test_init_stores_api_key() -> None:
    client = _TestClient(base_url="http://x", api_key="secret")
    assert client._api_key == "secret"


def test_init_api_key_defaults_to_none() -> None:
    client = _TestClient(base_url="http://x")
    assert client._api_key is None


def test_init_stores_request_timeout() -> None:
    client = _TestClient(base_url="http://x", request_timeout=3.5)
    assert client._request_timeout == 3.5


def test_init_request_timeout_defaults_to_none() -> None:
    client = _TestClient(base_url="http://x")
    assert client._request_timeout is None


# --------------------------------------------------------------------------- #
# _get — happy path
# --------------------------------------------------------------------------- #


def test_get_returns_parsed_json_dict(monkeypatch: pytest.MonkeyPatch) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"key": "value"})

    install_timeout_transport(monkeypatch, handler, _base_module)
    client = _TestClient(base_url="http://example.com/api")
    result = asyncio.run(client._get("/v1/test"))
    assert result == {"key": "value"}


def test_get_constructs_url_correctly(monkeypatch: pytest.MonkeyPatch) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        assert str(request.url) == "http://example.com/api/v1/test"
        return httpx.Response(200, json={"ok": True})

    install_timeout_transport(monkeypatch, handler, _base_module)
    client = _TestClient(base_url="http://example.com/api")
    asyncio.run(client._get("/v1/test"))


def test_get_passes_query_params(monkeypatch: pytest.MonkeyPatch) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.params["a"] == "1"
        assert request.url.params["b"] == "2"
        return httpx.Response(200, json={"ok": True})

    install_timeout_transport(monkeypatch, handler, _base_module)
    client = _TestClient(base_url="http://example.com/api")
    asyncio.run(client._get("/v1/test", params={"a": "1", "b": "2"}))


# --------------------------------------------------------------------------- #
# _get — authorization header
# --------------------------------------------------------------------------- #


def test_get_sets_authorization_header_when_api_key_provided(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.headers["Authorization"] == "Bearer my-key"
        return httpx.Response(200, json={"ok": True})

    install_timeout_transport(monkeypatch, handler, _base_module)
    client = _TestClient(base_url="http://example.com/api", api_key="my-key")
    asyncio.run(client._get("/v1/test"))


def test_get_omits_authorization_header_when_api_key_is_none(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        assert "Authorization" not in request.headers
        return httpx.Response(200, json={"ok": True})

    install_timeout_transport(monkeypatch, handler, _base_module)
    client = _TestClient(base_url="http://example.com/api", api_key=None)
    asyncio.run(client._get("/v1/test"))


# --------------------------------------------------------------------------- #
# _get — HTTP error status codes
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("status", [400, 404, 500, 503])
def test_get_raises_on_non_2xx_status(
    monkeypatch: pytest.MonkeyPatch, status: int
) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(status)

    install_timeout_transport(monkeypatch, handler, _base_module)
    client = _TestClient(base_url="http://example.com/api")
    with pytest.raises(_TestClientError, match=f"HTTP {status}"):
        asyncio.run(client._get("/v1/test"))


# --------------------------------------------------------------------------- #
# _get — non-JSON / unexpected JSON shape
# --------------------------------------------------------------------------- #


def test_get_raises_on_non_json_body(monkeypatch: pytest.MonkeyPatch) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, content=b"plain text, not json")

    install_timeout_transport(monkeypatch, handler, _base_module)
    client = _TestClient(base_url="http://example.com/api")
    with pytest.raises(_TestClientError, match="non-JSON body"):
        asyncio.run(client._get("/v1/test"))


def test_get_raises_on_json_array_body(monkeypatch: pytest.MonkeyPatch) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=["a", "b"])

    install_timeout_transport(monkeypatch, handler, _base_module)
    client = _TestClient(base_url="http://example.com/api")
    with pytest.raises(_TestClientError, match="unexpected JSON shape"):
        asyncio.run(client._get("/v1/test"))


def test_get_raises_on_json_string_body(monkeypatch: pytest.MonkeyPatch) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json="a string")

    install_timeout_transport(monkeypatch, handler, _base_module)
    client = _TestClient(base_url="http://example.com/api")
    with pytest.raises(_TestClientError, match="unexpected JSON shape"):
        asyncio.run(client._get("/v1/test"))


def test_get_raises_on_json_number_body(monkeypatch: pytest.MonkeyPatch) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=42)

    install_timeout_transport(monkeypatch, handler, _base_module)
    client = _TestClient(base_url="http://example.com/api")
    with pytest.raises(_TestClientError, match="unexpected JSON shape"):
        asyncio.run(client._get("/v1/test"))


# --------------------------------------------------------------------------- #
# _get — network / transport error
# --------------------------------------------------------------------------- #


def test_get_raises_on_network_error(monkeypatch: pytest.MonkeyPatch) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("connection refused")

    install_timeout_transport(monkeypatch, handler, _base_module)
    client = _TestClient(base_url="http://example.com/api")
    with pytest.raises(_TestClientError, match="connection refused"):
        asyncio.run(client._get("/v1/test"))


# --------------------------------------------------------------------------- #
# _get — request timeout
# --------------------------------------------------------------------------- #


def test_get_applies_request_timeout(monkeypatch: pytest.MonkeyPatch) -> None:
    captured_clients: list[httpx.AsyncClient] = []

    transport = httpx.MockTransport(lambda req: httpx.Response(200, json={"ok": True}))

    def _fake_timeout_client() -> httpx.AsyncClient:
        client = httpx.AsyncClient(transport=transport)
        captured_clients.append(client)
        return client

    monkeypatch.setattr(_base_module, "timeout_http_client", _fake_timeout_client)

    client = _TestClient(base_url="http://example.com/api", request_timeout=5.0)
    asyncio.run(client._get("/v1/test"))

    assert len(captured_clients) == 1
    assert isinstance(captured_clients[0].timeout, httpx.Timeout)
    assert captured_clients[0].timeout.read == 5.0


def test_get_does_not_set_timeout_when_request_timeout_is_none(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """When request_timeout is None, the client's default timeout is used."""
    captured_clients: list[httpx.AsyncClient] = []

    transport = httpx.MockTransport(lambda req: httpx.Response(200, json={"ok": True}))

    def _fake_timeout_client() -> httpx.AsyncClient:
        client = httpx.AsyncClient(transport=transport)
        captured_clients.append(client)
        return client

    monkeypatch.setattr(_base_module, "timeout_http_client", _fake_timeout_client)

    client = _TestClient(base_url="http://example.com/api", request_timeout=None)
    asyncio.run(client._get("/v1/test"))

    assert len(captured_clients) == 1
    # Default AsyncClient timeout is 5.0 seconds in all directions.
    assert captured_clients[0].timeout == httpx.Timeout(5.0)

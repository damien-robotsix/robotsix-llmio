"""Offline unit tests for ``AsyncOpenRouterClient``.

Drives the async OpenRouter REST client via ``httpx.MockTransport``
(no network, no respx dependency). Async methods are called via
``asyncio.run`` — no ``pytest-asyncio`` needed.
"""

from __future__ import annotations

import asyncio

import httpx
import pytest

from robotsix_llmio.openrouter import OpenRouterAPIError
from robotsix_llmio.openrouter import _async_client as _async_client_module
from robotsix_llmio.openrouter._async_client import AsyncOpenRouterClient
from robotsix_llmio.openrouter.provider_cost import KeyUsage
from tests.core.conftest import install_timeout_transport

# --------------------------------------------------------------------------- #
# fetch_key_usage
# --------------------------------------------------------------------------- #


def test_fetch_key_usage_parses_usage_limit_label(monkeypatch):
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/api/v1/auth/key"
        assert request.headers["Authorization"] == "Bearer k"
        return httpx.Response(
            200,
            json={"data": {"usage": 12.5, "limit": 100.0, "label": "mykey"}},
        )

    install_timeout_transport(monkeypatch, handler, _async_client_module)

    client = AsyncOpenRouterClient(api_key="k")
    result = asyncio.run(client.fetch_key_usage())

    assert isinstance(result, KeyUsage)
    assert result.usage == 12.5
    assert result.limit == 100.0
    assert result.label == "mykey"


def test_fetch_key_usage_limit_none_is_unlimited(monkeypatch):
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={"data": {"usage": 5.0, "limit": None, "label": "nolimit"}},
        )

    install_timeout_transport(monkeypatch, handler, _async_client_module)

    client = AsyncOpenRouterClient(api_key="k")
    result = asyncio.run(client.fetch_key_usage())

    assert result.usage == 5.0
    assert result.limit is None
    assert result.label == "nolimit"


def test_fetch_key_usage_non_2xx_raises_runtime_error(monkeypatch):
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(401, json={"error": "unauthorized"})

    install_timeout_transport(monkeypatch, handler, _async_client_module)

    client = AsyncOpenRouterClient(api_key="k")
    with pytest.raises(OpenRouterAPIError, match="HTTP 401"):
        asyncio.run(client.fetch_key_usage())


def test_fetch_key_usage_defaults_missing_fields(monkeypatch):
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"data": {}})

    install_timeout_transport(monkeypatch, handler, _async_client_module)

    client = AsyncOpenRouterClient(api_key="k")
    result = asyncio.run(client.fetch_key_usage())

    assert result.usage == 0.0
    assert result.limit is None
    assert result.label is None


def test_fetch_key_usage_missing_data_key_defaults(monkeypatch):
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={})

    install_timeout_transport(monkeypatch, handler, _async_client_module)

    client = AsyncOpenRouterClient(api_key="k")
    result = asyncio.run(client.fetch_key_usage())

    assert result.usage == 0.0
    assert result.limit is None
    assert result.label is None


# --------------------------------------------------------------------------- #
# fetch_credits
# --------------------------------------------------------------------------- #


def test_fetch_credits_returns_rounded_values(monkeypatch):
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/api/v1/credits"
        assert request.headers["Authorization"] == "Bearer k"
        return httpx.Response(
            200,
            json={
                "data": {
                    "total_credits": 100.1234567,
                    "total_usage": 25.1234567,
                }
            },
        )

    install_timeout_transport(monkeypatch, handler, _async_client_module)

    client = AsyncOpenRouterClient(api_key="k")
    result = asyncio.run(client.fetch_credits())

    assert isinstance(result, dict)
    assert result["total_credits"] == round(100.1234567, 6)
    assert result["total_usage"] == round(25.1234567, 6)
    assert result["remaining"] == round(100.1234567 - 25.1234567, 6)
    assert result["remaining"] == result["total_credits"] - result["total_usage"]


def test_fetch_credits_defaults_missing_fields(monkeypatch):
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"data": {"total_credits": 10.0}})

    install_timeout_transport(monkeypatch, handler, _async_client_module)

    client = AsyncOpenRouterClient(api_key="k")
    result = asyncio.run(client.fetch_credits())

    assert result["total_credits"] == 10.0
    assert result["total_usage"] == 0.0
    assert result["remaining"] == 10.0


def test_fetch_credits_missing_data_defaults(monkeypatch):
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={})

    install_timeout_transport(monkeypatch, handler, _async_client_module)

    client = AsyncOpenRouterClient(api_key="k")
    result = asyncio.run(client.fetch_credits())

    assert result["total_credits"] == 0.0
    assert result["total_usage"] == 0.0
    assert result["remaining"] == 0.0


# --------------------------------------------------------------------------- #
# Non-JSON / JSON-array body guards
# --------------------------------------------------------------------------- #


def test_fetch_key_usage_html_body_raises_openrouter_error(monkeypatch):
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            content=b"<html>Bad Gateway</html>",
            headers={"content-type": "text/html"},
        )

    install_timeout_transport(monkeypatch, handler, _async_client_module)
    client = AsyncOpenRouterClient(api_key="k")
    with pytest.raises(OpenRouterAPIError, match="non-JSON"):
        asyncio.run(client.fetch_key_usage())


def test_fetch_key_usage_json_array_body_raises_openrouter_error(monkeypatch):
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=["unexpected", "array"])

    install_timeout_transport(monkeypatch, handler, _async_client_module)
    client = AsyncOpenRouterClient(api_key="k")
    with pytest.raises(OpenRouterAPIError, match="unexpected JSON shape"):
        asyncio.run(client.fetch_key_usage())


def test_fetch_credits_non_2xx_raises_runtime_error(monkeypatch):
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(403, json={"error": "forbidden"})

    install_timeout_transport(monkeypatch, handler, _async_client_module)

    client = AsyncOpenRouterClient(api_key="k")
    with pytest.raises(OpenRouterAPIError, match="HTTP 403"):
        asyncio.run(client.fetch_credits())

"""Offline unit tests for the refdocs module.

Drives the async refdocs REST client and tool factory via
``httpx.MockTransport`` (no network, no respx dependency). Async
methods are called via ``asyncio.run`` — no ``pytest-asyncio`` needed.
"""

from __future__ import annotations

import asyncio
import json

import httpx
import pytest

from robotsix_llmio.clients import _base as _base_module
from robotsix_llmio.clients.refdocs import RefdocsClientError
from robotsix_llmio.clients.refdocs._async_client import AsyncRefdocsClient
from robotsix_llmio.clients.refdocs._settings import RefdocsSettings
from robotsix_llmio.clients.refdocs.factory import build_refdocs_tools
from tests.core.conftest import install_timeout_transport

# --------------------------------------------------------------------------- #
#  AsyncRefdocsClient — search
# --------------------------------------------------------------------------- #


def test_search_returns_results(monkeypatch):
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/search"
        assert request.url.params["q"] == "how to configure"
        return httpx.Response(
            200,
            json={
                "results": [
                    {"path": "config/index", "title": "Configuration guide"},
                    {"path": "core/factory", "title": "Agent factory"},
                ]
            },
        )

    install_timeout_transport(monkeypatch, handler, _base_module)

    client = AsyncRefdocsClient(base_url="http://refdocs:9090")
    results = asyncio.run(client.search("how to configure"))

    assert len(results) == 2
    assert results[0]["path"] == "config/index"
    assert results[0]["title"] == "Configuration guide"
    assert results[1]["path"] == "core/factory"


def test_search_non_list_results_defaults_to_empty(monkeypatch):
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"results": None})

    install_timeout_transport(monkeypatch, handler, _base_module)

    client = AsyncRefdocsClient()
    results = asyncio.run(client.search("query"))
    assert results == []


def test_search_missing_results_key_defaults_to_empty(monkeypatch):
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={})

    install_timeout_transport(monkeypatch, handler, _base_module)

    client = AsyncRefdocsClient()
    results = asyncio.run(client.search("query"))
    assert results == []


def test_search_non_2xx_raises_refdocs_client_error(monkeypatch):
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(500, json={"error": "internal"})

    install_timeout_transport(monkeypatch, handler, _base_module)

    client = AsyncRefdocsClient()
    with pytest.raises(RefdocsClientError, match="HTTP 500"):
        asyncio.run(client.search("query"))


# --------------------------------------------------------------------------- #
#  AsyncRefdocsClient — get_doc
# --------------------------------------------------------------------------- #


def test_get_doc_returns_content(monkeypatch):
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/docs/core/agent"
        return httpx.Response(
            200,
            json={"content": "# Agent module\n\nThis is the agent documentation."},
        )

    install_timeout_transport(monkeypatch, handler, _base_module)

    client = AsyncRefdocsClient(base_url="http://refdocs:9090")
    content = asyncio.run(client.get_doc("core/agent"))

    assert content == "# Agent module\n\nThis is the agent documentation."


def test_get_doc_missing_content_defaults_to_empty(monkeypatch):
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={})

    install_timeout_transport(monkeypatch, handler, _base_module)

    client = AsyncRefdocsClient()
    content = asyncio.run(client.get_doc("nonexistent"))
    assert content == ""


def test_get_doc_non_2xx_raises_refdocs_client_error(monkeypatch):
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(404, json={"error": "not found"})

    install_timeout_transport(monkeypatch, handler, _base_module)

    client = AsyncRefdocsClient()
    with pytest.raises(RefdocsClientError, match="HTTP 404"):
        asyncio.run(client.get_doc("missing"))


# --------------------------------------------------------------------------- #
#  AsyncRefdocsClient — auth
# --------------------------------------------------------------------------- #


def test_api_key_adds_bearer_header(monkeypatch):
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.headers["Authorization"] == "Bearer secret-key"
        return httpx.Response(200, json={"results": []})

    install_timeout_transport(monkeypatch, handler, _base_module)

    client = AsyncRefdocsClient(api_key="secret-key")
    asyncio.run(client.search("q"))


def test_no_api_key_no_auth_header(monkeypatch):
    def handler(request: httpx.Request) -> httpx.Response:
        assert "Authorization" not in request.headers
        return httpx.Response(200, json={"results": []})

    install_timeout_transport(monkeypatch, handler, _base_module)

    client = AsyncRefdocsClient()
    asyncio.run(client.search("q"))


# --------------------------------------------------------------------------- #
#  Non-JSON / JSON-array body guards
# --------------------------------------------------------------------------- #


def test_search_html_body_raises_refdocs_error(monkeypatch):
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            content=b"<html>Bad Gateway</html>",
            headers={"content-type": "text/html"},
        )

    install_timeout_transport(monkeypatch, handler, _base_module)
    client = AsyncRefdocsClient(base_url="http://rd")
    with pytest.raises(RefdocsClientError, match="non-JSON"):
        asyncio.run(client.search("q"))


def test_search_json_array_body_raises_refdocs_error(monkeypatch):
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=["unexpected", "array"])

    install_timeout_transport(monkeypatch, handler, _base_module)
    client = AsyncRefdocsClient(base_url="http://rd")
    with pytest.raises(RefdocsClientError, match="unexpected JSON shape"):
        asyncio.run(client.search("q"))


# --------------------------------------------------------------------------- #
#  RefdocsSettings
# --------------------------------------------------------------------------- #


def test_settings_defaults():
    s = RefdocsSettings()
    assert s.base_url == "http://localhost:9090"
    assert s.api_key is None
    assert s.request_timeout == 30.0


def test_settings_explicit_api_key():
    s = RefdocsSettings(api_key="explicit-key")
    assert s.resolved_api_key == "explicit-key"


def test_settings_api_key_env_fallback(monkeypatch):
    monkeypatch.setenv("REFDOCS_API_KEY", "env-key")
    s = RefdocsSettings()
    assert s.resolved_api_key == "env-key"


def test_settings_explicit_wins_over_env(monkeypatch):
    monkeypatch.setenv("REFDOCS_API_KEY", "env-key")
    s = RefdocsSettings(api_key="explicit-key")
    assert s.resolved_api_key == "explicit-key"


def test_settings_no_api_key_returns_none(monkeypatch):
    monkeypatch.delenv("REFDOCS_API_KEY", raising=False)
    s = RefdocsSettings()
    assert s.resolved_api_key is None


# --------------------------------------------------------------------------- #
#  build_refdocs_tools
# --------------------------------------------------------------------------- #


def test_build_refdocs_tools_returns_two_tools(monkeypatch):
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"results": []})

    install_timeout_transport(monkeypatch, handler, _base_module)

    from pydantic_ai.tools import Tool

    settings = RefdocsSettings(base_url="http://refdocs:9090")
    tools = build_refdocs_tools(settings)

    assert len(tools) == 2
    assert all(isinstance(t, Tool) for t in tools)
    names = {t.name for t in tools}
    assert names == {"search_refdocs", "get_refdocs"}


def test_search_refdocs_tool_calls_search(monkeypatch):
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                "results": [
                    {"path": "a", "title": "A"},
                ]
            },
        )

    install_timeout_transport(monkeypatch, handler, _base_module)

    settings = RefdocsSettings(base_url="http://refdocs:9090")
    tools = build_refdocs_tools(settings)
    search_tool = next(t for t in tools if t.name == "search_refdocs")

    result = asyncio.run(search_tool.function(query="test"))
    parsed = json.loads(result)
    assert len(parsed) == 1
    assert parsed[0]["path"] == "a"


def test_get_refdocs_tool_calls_get_doc(monkeypatch):
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={"content": "doc body"},
        )

    install_timeout_transport(monkeypatch, handler, _base_module)

    settings = RefdocsSettings(base_url="http://refdocs:9090")
    tools = build_refdocs_tools(settings)
    get_tool = next(t for t in tools if t.name == "get_refdocs")

    result = asyncio.run(get_tool.function(path="some/doc"))
    assert result == "doc body"

"""Offline unit tests for the knowledge-store client and tool adapter.

Drives ``KnowledgeClient`` via ``httpx.MockTransport`` (no network,
no respx dependency). Async methods are called via ``asyncio.run`` —
no ``pytest-asyncio`` needed.
"""

from __future__ import annotations

import asyncio
from collections.abc import Callable

import httpx
import pytest

from robotsix_llmio.clients import _base as _base_module
from robotsix_llmio.clients.knowledge import KnowledgeClientError
from robotsix_llmio.clients.knowledge._client import (
    KnowledgeClient,
    build_knowledge_tools,
)

# --------------------------------------------------------------------------- #
# Test helpers
# --------------------------------------------------------------------------- #


def _install_transport(
    monkeypatch: pytest.MonkeyPatch,
    handler: Callable[[httpx.Request], httpx.Response],
) -> None:
    """Patch ``timeout_http_client`` so the client under test uses a
    ``MockTransport`` running *handler*."""
    transport = httpx.MockTransport(handler)

    def _fake_timeout_client() -> httpx.AsyncClient:
        return httpx.AsyncClient(transport=transport)

    monkeypatch.setattr(_base_module, "timeout_http_client", _fake_timeout_client)


# --------------------------------------------------------------------------- #
# KnowledgeClient.search
# --------------------------------------------------------------------------- #


def test_search_sends_query_and_top_k(monkeypatch: pytest.MonkeyPatch) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/api/v1/search"
        # httpx stores params as a QueryParams on request.url
        assert "q" in str(request.url.query)
        assert "top_k" in str(request.url.query)
        return httpx.Response(200, json={"results": []})

    _install_transport(monkeypatch, handler)

    client = KnowledgeClient(base_url="http://ks:8000/api/v1")
    results = asyncio.run(client.search("test query", top_k=5))
    assert results == []


def test_search_parses_results(monkeypatch: pytest.MonkeyPatch) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                "results": [
                    {
                        "id": "doc-1",
                        "title": "Alpha",
                        "snippet": "First doc",
                        "score": 0.95,
                    },
                    {
                        "id": "doc-2",
                        "title": "Beta",
                        "snippet": "Second doc",
                        "score": 0.72,
                    },
                ]
            },
        )

    _install_transport(monkeypatch, handler)

    client = KnowledgeClient(base_url="http://ks:8000/api/v1")
    results = asyncio.run(client.search("anything"))

    assert len(results) == 2
    assert results[0]["id"] == "doc-1"
    assert results[0]["title"] == "Alpha"
    assert results[0]["snippet"] == "First doc"
    assert results[0]["score"] == 0.95
    assert results[1]["id"] == "doc-2"


def test_search_non_2xx_raises_knowledge_client_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(500, json={"error": "internal"})

    _install_transport(monkeypatch, handler)

    client = KnowledgeClient(base_url="http://ks:8000/api/v1")
    with pytest.raises(KnowledgeClientError, match="HTTP 500"):
        asyncio.run(client.search("boom"))


def test_search_network_error_raises_knowledge_client_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Simulate a transport-level failure (e.g. connection refused)."""

    # Replace timeout_http_client with a fake that raises inside the context
    # manager's __aenter__ (i.e. when the async with block tries to create
    # the client).
    def _fake_timeout_client() -> httpx.AsyncClient:
        raise ConnectionError("connection refused")

    monkeypatch.setattr(_base_module, "timeout_http_client", _fake_timeout_client)

    client = KnowledgeClient(base_url="http://ks:8000/api/v1")
    with pytest.raises(KnowledgeClientError, match="connection refused"):
        asyncio.run(client.search("query"))


def test_search_includes_api_key(monkeypatch: pytest.MonkeyPatch) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.headers["Authorization"] == "Bearer secret-key"
        return httpx.Response(200, json={"results": []})

    _install_transport(monkeypatch, handler)

    client = KnowledgeClient(base_url="http://ks:8000/api/v1", api_key="secret-key")
    asyncio.run(client.search("query"))


def test_search_no_api_key_omits_auth_header(monkeypatch: pytest.MonkeyPatch) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        assert "Authorization" not in request.headers
        return httpx.Response(200, json={"results": []})

    _install_transport(monkeypatch, handler)

    client = KnowledgeClient(base_url="http://ks:8000/api/v1")
    asyncio.run(client.search("query"))


def test_search_missing_results_key_defaults_to_empty(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={})

    _install_transport(monkeypatch, handler)

    client = KnowledgeClient(base_url="http://ks:8000/api/v1")
    results = asyncio.run(client.search("query"))
    assert results == []


# --------------------------------------------------------------------------- #
# KnowledgeClient.get_document
# --------------------------------------------------------------------------- #


def test_get_document_returns_full_doc(monkeypatch: pytest.MonkeyPatch) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/api/v1/documents/doc-42"
        return httpx.Response(
            200,
            json={
                "id": "doc-42",
                "title": "The Answer",
                "content": "It's 42.",
                "metadata": {"source": "guide"},
            },
        )

    _install_transport(monkeypatch, handler)

    client = KnowledgeClient(base_url="http://ks:8000/api/v1")
    doc = asyncio.run(client.get_document("doc-42"))

    assert doc["id"] == "doc-42"
    assert doc["title"] == "The Answer"
    assert doc["content"] == "It's 42."
    assert doc["metadata"] == {"source": "guide"}


def test_get_document_404_raises_knowledge_client_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(404, json={"error": "not found"})

    _install_transport(monkeypatch, handler)

    client = KnowledgeClient(base_url="http://ks:8000/api/v1")
    with pytest.raises(KnowledgeClientError, match="HTTP 404"):
        asyncio.run(client.get_document("missing"))


def test_get_document_includes_api_key(monkeypatch: pytest.MonkeyPatch) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.headers["Authorization"] == "Bearer tok"
        return httpx.Response(200, json={"id": "x", "title": "T", "content": "C"})

    _install_transport(monkeypatch, handler)

    client = KnowledgeClient(base_url="http://ks:8000/api/v1", api_key="tok")
    asyncio.run(client.get_document("x"))


# --------------------------------------------------------------------------- #
# build_knowledge_tools
# --------------------------------------------------------------------------- #


def test_build_knowledge_tools_returns_two_functions() -> None:
    client = KnowledgeClient(base_url="http://ks:8000/api/v1")
    tools = build_knowledge_tools(client)
    assert len(tools) == 2
    assert all(callable(t) for t in tools)


def test_search_knowledge_tool_returns_formatted_results(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                "results": [
                    {
                        "id": "doc-1",
                        "title": "Intro",
                        "snippet": "Welcome...",
                        "score": 0.99,
                    },
                ]
            },
        )

    _install_transport(monkeypatch, handler)

    client = KnowledgeClient(base_url="http://ks:8000/api/v1")
    tools = build_knowledge_tools(client)
    search_fn = tools[0]

    result = asyncio.run(search_fn("hello"))
    assert "doc-1" in result
    assert "Intro" in result
    assert "score=0.99" in result
    assert "Welcome..." in result


def test_search_knowledge_tool_no_results(monkeypatch: pytest.MonkeyPatch) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"results": []})

    _install_transport(monkeypatch, handler)

    client = KnowledgeClient(base_url="http://ks:8000/api/v1")
    tools = build_knowledge_tools(client)
    search_fn = tools[0]

    result = asyncio.run(search_fn("nonesuch"))
    assert "No documents found" in result


def test_search_knowledge_tool_handles_error(monkeypatch: pytest.MonkeyPatch) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(503, json={"error": "unavailable"})

    _install_transport(monkeypatch, handler)

    client = KnowledgeClient(base_url="http://ks:8000/api/v1")
    tools = build_knowledge_tools(client)
    search_fn = tools[0]

    result = asyncio.run(search_fn("query"))
    assert "Search failed" in result
    assert "503" in result


def test_get_knowledge_document_tool_returns_formatted_doc(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                "id": "doc-7",
                "title": "Secrets",
                "content": "The secret is 7.",
            },
        )

    _install_transport(monkeypatch, handler)

    client = KnowledgeClient(base_url="http://ks:8000/api/v1")
    tools = build_knowledge_tools(client)
    fetch_fn = tools[1]

    result = asyncio.run(fetch_fn("doc-7"))
    assert "# Secrets" in result
    assert "The secret is 7." in result


def test_get_knowledge_document_tool_empty_content(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={"id": "empty", "title": "Nothing", "content": ""},
        )

    _install_transport(monkeypatch, handler)

    client = KnowledgeClient(base_url="http://ks:8000/api/v1")
    tools = build_knowledge_tools(client)
    fetch_fn = tools[1]

    result = asyncio.run(fetch_fn("empty"))
    assert "has no content" in result


def test_get_knowledge_document_tool_handles_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(404, json={"error": "not found"})

    _install_transport(monkeypatch, handler)

    client = KnowledgeClient(base_url="http://ks:8000/api/v1")
    tools = build_knowledge_tools(client)
    fetch_fn = tools[1]

    result = asyncio.run(fetch_fn("ghost"))
    assert "Failed to retrieve document ghost" in result
    assert "404" in result


# --------------------------------------------------------------------------- #
# Non-JSON / JSON-array body guards
# --------------------------------------------------------------------------- #


def test_search_html_body_raises_knowledge_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            content=b"<html>Bad Gateway</html>",
            headers={"content-type": "text/html"},
        )

    _install_transport(monkeypatch, handler)
    client = KnowledgeClient(base_url="http://ks:8000/api/v1")
    with pytest.raises(KnowledgeClientError, match="non-JSON"):
        asyncio.run(client.search("q"))


def test_search_json_array_body_raises_knowledge_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=["unexpected", "array"])

    _install_transport(monkeypatch, handler)
    client = KnowledgeClient(base_url="http://ks:8000/api/v1")
    with pytest.raises(KnowledgeClientError, match="unexpected JSON shape"):
        asyncio.run(client.search("q"))


# --------------------------------------------------------------------------- #
# Error hierarchy
# --------------------------------------------------------------------------- #


def test_knowledge_client_error_is_robotsix_llmio_error() -> None:
    from robotsix_llmio.exceptions import RobotsixLLMIOError

    assert issubclass(KnowledgeClientError, RobotsixLLMIOError)


def test_knowledge_client_error_can_be_caught_as_base() -> None:
    from robotsix_llmio.exceptions import RobotsixLLMIOError

    try:
        raise KnowledgeClientError("test")
    except RobotsixLLMIOError:
        pass  # expected
    else:
        pytest.fail("KnowledgeClientError should be catchable as RobotsixLLMIOError")

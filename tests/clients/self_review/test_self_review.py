"""Offline unit tests for the self-review client and tool adapter.

Drives ``SelfReviewClient`` via ``httpx.MockTransport`` (no network,
no respx dependency). Async methods are called via ``asyncio.run`` —
no ``pytest-asyncio`` needed.
"""

from __future__ import annotations

import asyncio
from collections.abc import Callable

import httpx
import pytest

from robotsix_llmio.clients import _base as _base_module
from robotsix_llmio.clients.self_review import SelfReviewClientError
from robotsix_llmio.clients.self_review._client import (
    SelfReviewClient,
    build_recent_activity_tools,
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
# SelfReviewClient.list_activity
# --------------------------------------------------------------------------- #


def test_list_activity_sends_limit(monkeypatch: pytest.MonkeyPatch) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/api/v1/activity"
        assert "limit" in str(request.url.query)
        return httpx.Response(200, json={"activities": []})

    _install_transport(monkeypatch, handler)

    client = SelfReviewClient(base_url="http://sr:8000/api/v1")
    activities = asyncio.run(client.list_activity(limit=5))
    assert activities == []


def test_list_activity_parses_results(monkeypatch: pytest.MonkeyPatch) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                "activities": [
                    {
                        "id": "act-1",
                        "agent": "planner",
                        "action": "plan",
                        "summary": "Planned sprint",
                        "timestamp": "2026-06-29T10:00:00Z",
                    },
                    {
                        "id": "act-2",
                        "agent": "implement",
                        "action": "code",
                        "summary": "Wrote tests",
                        "timestamp": "2026-06-29T11:00:00Z",
                    },
                ]
            },
        )

    _install_transport(monkeypatch, handler)

    client = SelfReviewClient(base_url="http://sr:8000/api/v1")
    activities = asyncio.run(client.list_activity())

    assert len(activities) == 2
    assert activities[0]["id"] == "act-1"
    assert activities[0]["agent"] == "planner"
    assert activities[0]["action"] == "plan"
    assert activities[0]["summary"] == "Planned sprint"
    assert activities[0]["timestamp"] == "2026-06-29T10:00:00Z"
    assert activities[1]["id"] == "act-2"


def test_list_activity_non_2xx_raises_self_review_client_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(500, json={"error": "internal"})

    _install_transport(monkeypatch, handler)

    client = SelfReviewClient(base_url="http://sr:8000/api/v1")
    with pytest.raises(SelfReviewClientError, match="HTTP 500"):
        asyncio.run(client.list_activity())


def test_list_activity_network_error_raises_self_review_client_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Simulate a transport-level failure (e.g. connection refused)."""

    def _fake_timeout_client() -> httpx.AsyncClient:
        raise ConnectionError("connection refused")

    monkeypatch.setattr(_base_module, "timeout_http_client", _fake_timeout_client)

    client = SelfReviewClient(base_url="http://sr:8000/api/v1")
    with pytest.raises(SelfReviewClientError, match="connection refused"):
        asyncio.run(client.list_activity())


def test_list_activity_includes_api_key(monkeypatch: pytest.MonkeyPatch) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.headers["Authorization"] == "Bearer secret-key"
        return httpx.Response(200, json={"activities": []})

    _install_transport(monkeypatch, handler)

    client = SelfReviewClient(base_url="http://sr:8000/api/v1", api_key="secret-key")
    asyncio.run(client.list_activity())


def test_list_activity_no_api_key_omits_auth_header(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        assert "Authorization" not in request.headers
        return httpx.Response(200, json={"activities": []})

    _install_transport(monkeypatch, handler)

    client = SelfReviewClient(base_url="http://sr:8000/api/v1")
    asyncio.run(client.list_activity())


def test_list_activity_missing_activities_key_defaults_to_empty(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={})

    _install_transport(monkeypatch, handler)

    client = SelfReviewClient(base_url="http://sr:8000/api/v1")
    activities = asyncio.run(client.list_activity())
    assert activities == []


# --------------------------------------------------------------------------- #
# SelfReviewClient.get_activity
# --------------------------------------------------------------------------- #


def test_get_activity_returns_full_activity(monkeypatch: pytest.MonkeyPatch) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/api/v1/activity/act-42"
        return httpx.Response(
            200,
            json={
                "id": "act-42",
                "agent": "planner",
                "action": "plan",
                "summary": "Planned next sprint",
                "timestamp": "2026-06-29T12:00:00Z",
                "detail": "Detailed plan here.",
            },
        )

    _install_transport(monkeypatch, handler)

    client = SelfReviewClient(base_url="http://sr:8000/api/v1")
    activity = asyncio.run(client.get_activity("act-42"))

    assert activity["id"] == "act-42"
    assert activity["agent"] == "planner"
    assert activity["action"] == "plan"
    assert activity["summary"] == "Planned next sprint"
    assert activity["timestamp"] == "2026-06-29T12:00:00Z"
    assert activity["detail"] == "Detailed plan here."


def test_get_activity_404_raises_self_review_client_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(404, json={"error": "not found"})

    _install_transport(monkeypatch, handler)

    client = SelfReviewClient(base_url="http://sr:8000/api/v1")
    with pytest.raises(SelfReviewClientError, match="HTTP 404"):
        asyncio.run(client.get_activity("missing"))


def test_get_activity_includes_api_key(monkeypatch: pytest.MonkeyPatch) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.headers["Authorization"] == "Bearer tok"
        return httpx.Response(
            200,
            json={
                "id": "x",
                "agent": "a",
                "action": "b",
                "summary": "s",
                "timestamp": "t",
            },
        )

    _install_transport(monkeypatch, handler)

    client = SelfReviewClient(base_url="http://sr:8000/api/v1", api_key="tok")
    asyncio.run(client.get_activity("x"))


# --------------------------------------------------------------------------- #
# Non-JSON / JSON-array body guards
# --------------------------------------------------------------------------- #


def test_list_activity_html_body_raises_self_review_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            content=b"<html>Bad Gateway</html>",
            headers={"content-type": "text/html"},
        )

    _install_transport(monkeypatch, handler)
    client = SelfReviewClient(base_url="http://sr:8000/api/v1")
    with pytest.raises(SelfReviewClientError, match="non-JSON"):
        asyncio.run(client.list_activity())


def test_list_activity_json_array_body_raises_self_review_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=["unexpected", "array"])

    _install_transport(monkeypatch, handler)
    client = SelfReviewClient(base_url="http://sr:8000/api/v1")
    with pytest.raises(SelfReviewClientError, match="unexpected JSON shape"):
        asyncio.run(client.list_activity())


# --------------------------------------------------------------------------- #
# build_recent_activity_tools
# --------------------------------------------------------------------------- #


def test_build_recent_activity_tools_returns_two_functions() -> None:
    client = SelfReviewClient(base_url="http://sr:8000/api/v1")
    tools = build_recent_activity_tools(client)
    assert len(tools) == 2
    assert all(callable(t) for t in tools)


def test_list_recent_activity_tool_returns_formatted_results(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                "activities": [
                    {
                        "id": "act-1",
                        "agent": "implement",
                        "action": "code",
                        "summary": "Built module",
                        "timestamp": "2026-06-29T10:00:00Z",
                    },
                ]
            },
        )

    _install_transport(monkeypatch, handler)

    client = SelfReviewClient(base_url="http://sr:8000/api/v1")
    tools = build_recent_activity_tools(client)
    list_fn = tools[0]

    result = asyncio.run(list_fn(limit=10))
    assert "act-1" in result
    assert "implement" in result
    assert "code" in result
    assert "Built module" in result


def test_list_recent_activity_tool_no_results(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"activities": []})

    _install_transport(monkeypatch, handler)

    client = SelfReviewClient(base_url="http://sr:8000/api/v1")
    tools = build_recent_activity_tools(client)
    list_fn = tools[0]

    result = asyncio.run(list_fn())
    assert "No recent activity found" in result


def test_list_recent_activity_tool_handles_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(503, json={"error": "unavailable"})

    _install_transport(monkeypatch, handler)

    client = SelfReviewClient(base_url="http://sr:8000/api/v1")
    tools = build_recent_activity_tools(client)
    list_fn = tools[0]

    result = asyncio.run(list_fn())
    assert "List activity failed" in result
    assert "503" in result


def test_get_recent_activity_detail_tool_returns_formatted_activity(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                "id": "act-7",
                "agent": "planner",
                "action": "plan",
                "summary": "Planned migration",
                "timestamp": "2026-06-29T09:00:00Z",
                "detail": "Step 1: ...\nStep 2: ...",
            },
        )

    _install_transport(monkeypatch, handler)

    client = SelfReviewClient(base_url="http://sr:8000/api/v1")
    tools = build_recent_activity_tools(client)
    fetch_fn = tools[1]

    result = asyncio.run(fetch_fn("act-7"))
    assert "Activity act-7" in result
    assert "planner" in result
    assert "Planned migration" in result
    assert "Step 1:" in result
    assert "Step 2:" in result


def test_get_recent_activity_detail_tool_no_detail_field(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                "id": "act-8",
                "agent": "implement",
                "action": "code",
                "summary": "Quick fix",
                "timestamp": "2026-06-29T08:00:00Z",
            },
        )

    _install_transport(monkeypatch, handler)

    client = SelfReviewClient(base_url="http://sr:8000/api/v1")
    tools = build_recent_activity_tools(client)
    fetch_fn = tools[1]

    result = asyncio.run(fetch_fn("act-8"))
    assert "Activity act-8" in result
    assert "Quick fix" in result
    # No detail block since detail is absent


def test_get_recent_activity_detail_tool_handles_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(404, json={"error": "not found"})

    _install_transport(monkeypatch, handler)

    client = SelfReviewClient(base_url="http://sr:8000/api/v1")
    tools = build_recent_activity_tools(client)
    fetch_fn = tools[1]

    result = asyncio.run(fetch_fn("ghost"))
    assert "Failed to retrieve activity ghost" in result
    assert "404" in result


# --------------------------------------------------------------------------- #
# build_recent_activity_tools — conversation_store parameter
# --------------------------------------------------------------------------- #


def test_build_recent_activity_tools_accepts_conversation_store() -> None:
    """The conversation_store parameter is accepted (future-proofing)."""
    client = SelfReviewClient(base_url="http://sr:8000/api/v1")
    store = object()
    tools = build_recent_activity_tools(client, conversation_store=store)
    assert len(tools) == 2


# --------------------------------------------------------------------------- #
# Error hierarchy
# --------------------------------------------------------------------------- #


def test_self_review_client_error_is_robotsix_llmio_error() -> None:
    from robotsix_llmio.exceptions import RobotsixLLMIOError

    assert issubclass(SelfReviewClientError, RobotsixLLMIOError)


def test_self_review_client_error_can_be_caught_as_base() -> None:
    from robotsix_llmio.exceptions import RobotsixLLMIOError

    try:
        raise SelfReviewClientError("test")
    except RobotsixLLMIOError:
        pass  # expected
    else:
        pytest.fail("SelfReviewClientError should be catchable as RobotsixLLMIOError")

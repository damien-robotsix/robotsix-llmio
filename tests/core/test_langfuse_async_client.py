"""Unit tests for the reusable ``AsyncLangfuseReadClient`` kernel.

Drives the Langfuse REST async read client offline via
``httpx.MockTransport``, covering the auth header, base-URL resolution,
``aiter_pages`` pagination, ``fetch_traces_window``, ``fetch_trace_detail``,
and the static parsing helpers.
"""

from __future__ import annotations

import asyncio
import base64
from datetime import UTC, datetime

import httpx
import pytest
from conftest import install_async_transport

from robotsix_llmio.core import (
    langfuse_async_client as langfuse_async_client_module,
)
from robotsix_llmio.core import langfuse_client as langfuse_client_module
from robotsix_llmio.core.langfuse_async_client import AsyncLangfuseReadClient


def _client() -> AsyncLangfuseReadClient:
    return AsyncLangfuseReadClient(
        public_key="pub", secret_key="sec", base_url="https://lf.example.com"
    )


# --------------------------------------------------------------------------- #
# REST read-path constants (wire contract)
# --------------------------------------------------------------------------- #
def test_rest_path_constants_match_wire_strings():
    assert langfuse_async_client_module._TRACES_PATH == "/api/public/traces"
    assert langfuse_client_module._OBSERVATIONS_PATH == "/api/public/observations"


# --------------------------------------------------------------------------- #
# auth_header / base_url / url
# --------------------------------------------------------------------------- #
def test_auth_header_is_basic_base64():
    expected = base64.b64encode(b"pub:sec").decode()
    assert _client().auth_header() == f"Basic {expected}"


def test_base_url_default_and_strip():
    assert (
        AsyncLangfuseReadClient(public_key="p", secret_key="s").base_url
        == "https://cloud.langfuse.com"
    )
    trimmed = AsyncLangfuseReadClient(
        public_key="p", secret_key="s", base_url="https://lf.example.com/"
    )
    assert trimmed.base_url == "https://lf.example.com"


def test_url_joins_path():
    client = _client()
    assert client.url("/api/public/traces") == (
        "https://lf.example.com/api/public/traces"
    )
    assert client.url("api/public/observations") == (
        "https://lf.example.com/api/public/observations"
    )


# --------------------------------------------------------------------------- #
# aiter_pages
# --------------------------------------------------------------------------- #
def test_aiter_pages_relative_path_and_auth(monkeypatch):
    """A relative path resolves against base_url and carries the Basic header."""

    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/api/public/traces"
        assert request.headers["Authorization"].startswith("Basic ")
        page = int(request.url.params["page"])
        data = [{"id": "t1"}] if page == 1 else []
        return httpx.Response(200, json={"data": data})

    captured = install_async_transport(
        monkeypatch, handler, module=langfuse_async_client_module
    )

    async def _run():
        pages = []
        async for p in _client().aiter_pages("/api/public/traces"):
            pages.append(p)
        return pages

    pages = asyncio.run(_run())
    assert pages == [[{"id": "t1"}]]
    assert [int(r.url.params["page"]) for r in captured] == [1, 2]


def test_aiter_pages_stops_on_total_pages(monkeypatch):
    """``meta.totalPages`` terminates the loop without an extra empty fetch."""
    pages = {
        1: {"data": [{"id": "a"}], "meta": {"totalPages": 2}},
        2: {"data": [{"id": "b"}], "meta": {"totalPages": 2}},
    }

    def handler(request: httpx.Request) -> httpx.Response:
        page = int(request.url.params["page"])
        return httpx.Response(200, json=pages[page])

    captured = install_async_transport(
        monkeypatch, handler, module=langfuse_async_client_module
    )

    async def _run():
        result = []
        async for p in _client().aiter_pages(
            "https://lf.example.com/api/public/traces"
        ):
            result.append(p)
        return result

    result = asyncio.run(_run())
    assert result == [[{"id": "a"}], [{"id": "b"}]]
    assert [int(r.url.params["page"]) for r in captured] == [1, 2]


def test_aiter_pages_non_2xx_raises(monkeypatch):
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(500, text="boom")

    install_async_transport(monkeypatch, handler, module=langfuse_async_client_module)

    async def _run():
        async for _ in _client().aiter_pages(
            "/api/public/traces", error_label="traces request"
        ):
            pass

    with pytest.raises(RuntimeError, match="traces request"):
        asyncio.run(_run())


# --------------------------------------------------------------------------- #
# fetch_traces_window
# --------------------------------------------------------------------------- #
def test_fetch_traces_window_yields_individual_traces(monkeypatch):
    """``fetch_traces_window`` flattens paginated pages and sets
    ``fromTimestamp``."""
    pages = {
        1: {
            "data": [{"id": "t1"}, {"id": "t2"}],
            "meta": {"totalPages": 2},
        },
        2: {"data": [{"id": "t3"}], "meta": {"totalPages": 2}},
    }

    def handler(request: httpx.Request) -> httpx.Response:
        page = int(request.url.params["page"])
        # Verify fromTimestamp was set
        assert "fromTimestamp" in request.url.params
        return httpx.Response(200, json=pages[page])

    captured = install_async_transport(
        monkeypatch, handler, module=langfuse_async_client_module
    )

    async def _run():
        traces = []
        async for t in _client().fetch_traces_window(hours=24):
            traces.append(t)
        return traces

    traces = asyncio.run(_run())
    assert traces == [{"id": "t1"}, {"id": "t2"}, {"id": "t3"}]
    assert len(captured) == 2


# --------------------------------------------------------------------------- #
# fetch_trace_detail
# --------------------------------------------------------------------------- #
def test_fetch_trace_detail_returns_json(monkeypatch):
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/api/public/traces/trace-42"
        return httpx.Response(200, json={"id": "trace-42", "name": "test"})

    captured = install_async_transport(
        monkeypatch, handler, module=langfuse_async_client_module
    )

    result = asyncio.run(_client().fetch_trace_detail("trace-42"))
    assert result == {"id": "trace-42", "name": "test"}
    assert len(captured) == 1


def test_fetch_trace_detail_non_2xx_raises(monkeypatch):
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(404, text="not found")

    install_async_transport(monkeypatch, handler, module=langfuse_async_client_module)

    with pytest.raises(RuntimeError, match="trace detail failed"):
        asyncio.run(_client().fetch_trace_detail("missing"))


# --------------------------------------------------------------------------- #
# static parsing helpers
# --------------------------------------------------------------------------- #
def test_parse_timestamp_z_suffix():
    assert AsyncLangfuseReadClient.parse_timestamp("2024-01-01T12:00:00Z") == datetime(
        2024, 1, 1, 12, 0, tzinfo=UTC
    )


def test_observation_provider_and_cost():
    assert (
        AsyncLangfuseReadClient.observation_provider(
            {"metadata": {"provider": "openrouter"}}
        )
        == "openrouter"
    )
    assert AsyncLangfuseReadClient.observation_provider({}) is None
    assert AsyncLangfuseReadClient.observation_cost(
        {"calculatedTotalCost": 1.0, "totalCost": 2.0}
    ) == pytest.approx(1.0)
    assert AsyncLangfuseReadClient.observation_cost({}) == 0.0

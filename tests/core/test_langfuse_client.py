"""Unit tests for the reusable ``LangfuseReadClient`` kernel.

Drives the Langfuse REST read client offline via ``httpx.MockTransport``,
covering the auth header, base-URL resolution, ``iter_pages`` pagination, and
the static parsing helpers (timestamp / observation provider / observation
cost).
"""

from __future__ import annotations

import base64
from datetime import UTC, datetime

import httpx
import pytest

from robotsix_llmio.core import langfuse_client as langfuse_client_module
from robotsix_llmio.core.langfuse_client import LangfuseReadClient


def _install_transport(monkeypatch, handler) -> list[httpx.Request]:
    """Patch ``httpx.Client`` so the client uses a ``MockTransport`` running
    *handler*. Returns a list that captures every request sent."""
    captured: list[httpx.Request] = []

    def _handler(request: httpx.Request) -> httpx.Response:
        captured.append(request)
        return handler(request)

    transport = httpx.MockTransport(_handler)
    real_client = httpx.Client

    def _client(*args, **kwargs):
        kwargs["transport"] = transport
        return real_client(*args, **kwargs)

    monkeypatch.setattr(langfuse_client_module.httpx, "Client", _client)
    return captured


def _client() -> LangfuseReadClient:
    return LangfuseReadClient(
        public_key="pub", secret_key="sec", base_url="https://lf.example.com"
    )


# --------------------------------------------------------------------------- #
# auth_header / base_url / url
# --------------------------------------------------------------------------- #
def test_auth_header_is_basic_base64():
    expected = base64.b64encode(b"pub:sec").decode()
    assert _client().auth_header() == f"Basic {expected}"


def test_base_url_default_and_strip():
    assert LangfuseReadClient(public_key="p", secret_key="s").base_url == (
        "https://cloud.langfuse.com"
    )
    trimmed = LangfuseReadClient(
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
# iter_pages
# --------------------------------------------------------------------------- #
def test_iter_pages_relative_path_and_auth(monkeypatch):
    """A relative path resolves against base_url and carries the Basic header."""

    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/api/public/traces"
        assert request.headers["Authorization"].startswith("Basic ")
        page = int(request.url.params["page"])
        data = [{"id": "t1"}] if page == 1 else []
        return httpx.Response(200, json={"data": data})

    captured = _install_transport(monkeypatch, handler)
    pages = list(_client().iter_pages("/api/public/traces"))

    assert pages == [[{"id": "t1"}]]
    assert [int(r.url.params["page"]) for r in captured] == [1, 2]


def test_iter_pages_stops_on_total_pages(monkeypatch):
    """``meta.totalPages`` terminates the loop without an extra empty fetch."""
    pages = {
        1: {"data": [{"id": "a"}], "meta": {"totalPages": 2}},
        2: {"data": [{"id": "b"}], "meta": {"totalPages": 2}},
    }

    def handler(request: httpx.Request) -> httpx.Response:
        page = int(request.url.params["page"])
        return httpx.Response(200, json=pages[page])

    captured = _install_transport(monkeypatch, handler)
    result = list(_client().iter_pages("https://lf.example.com/api/public/traces"))

    assert result == [[{"id": "a"}], [{"id": "b"}]]
    assert [int(r.url.params["page"]) for r in captured] == [1, 2]


def test_iter_pages_non_2xx_raises(monkeypatch):
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(500, text="boom")

    _install_transport(monkeypatch, handler)
    with pytest.raises(RuntimeError, match="traces request"):
        list(_client().iter_pages("/api/public/traces", error_label="traces request"))


# --------------------------------------------------------------------------- #
# static parsing helpers
# --------------------------------------------------------------------------- #
def test_parse_timestamp_z_suffix():
    assert LangfuseReadClient.parse_timestamp("2024-01-01T12:00:00Z") == datetime(
        2024, 1, 1, 12, 0, tzinfo=UTC
    )


def test_observation_provider_and_cost():
    assert (
        LangfuseReadClient.observation_provider(
            {"metadata": {"provider": "openrouter"}}
        )
        == "openrouter"
    )
    assert LangfuseReadClient.observation_provider({}) is None
    assert LangfuseReadClient.observation_cost(
        {"calculatedTotalCost": 1.0, "totalCost": 2.0}
    ) == pytest.approx(1.0)
    assert LangfuseReadClient.observation_cost({}) == 0.0

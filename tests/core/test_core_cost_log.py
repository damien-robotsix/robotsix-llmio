"""Langfuse read seam — ``LangfuseCostLogSource`` request building, pagination,
aggregation, and protocol conformance, driven offline via ``httpx.MockTransport``.
"""

from __future__ import annotations

import base64
from datetime import UTC, datetime

import httpx
import pytest
from conftest import install_transport, make_adapter, make_window

from robotsix_llmio.core.cost_log import CostLogSource, LoggedCost
from robotsix_llmio.core.langfuse_client import LangfuseClientError


def test_multi_page_aggregation(monkeypatch):
    pages = {
        1: [
            {"id": "t1", "totalCost": 0.5, "timestamp": "2026-06-03T10:01:00Z"},
            {"id": "t2", "totalCost": 1.25, "timestamp": "2026-06-03T10:02:00Z"},
        ],
        2: [
            {"id": "t3", "totalCost": 0.25, "timestamp": "2026-06-03T10:03:00Z"},
        ],
        3: [],
    }

    def handler(request: httpx.Request) -> httpx.Response:
        page = int(request.url.params["page"])
        return httpx.Response(200, json={"data": pages[page]})

    install_transport(monkeypatch, handler)
    result = make_adapter().fetch_logged_cost(make_window())

    assert isinstance(result, LoggedCost)
    assert result.record_count == 3
    assert result.total_cost == pytest.approx(2.0)


def test_per_record_population(monkeypatch):
    data = [
        {
            "id": "t1",
            "totalCost": 0.5,
            "timestamp": "2026-06-03T10:01:00Z",
            "sessionId": "sess-1",
            "name": "trace-one",
        },
    ]

    def handler(request: httpx.Request) -> httpx.Response:
        page = int(request.url.params["page"])
        return httpx.Response(200, json={"data": data if page == 1 else []})

    install_transport(monkeypatch, handler)
    result = make_adapter().fetch_logged_cost(make_window())

    assert len(result.records) == 1
    record = result.records[0]
    assert record.id == "t1"
    assert record.cost == pytest.approx(0.5)
    assert record.timestamp == datetime(2026, 6, 3, 10, 1, tzinfo=UTC)
    assert record.session_id == "sess-1"
    assert record.name == "trace-one"


def test_empty_window_zero_result(monkeypatch):
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"data": []})

    install_transport(monkeypatch, handler)
    result = make_adapter().fetch_logged_cost(make_window())

    assert result == LoggedCost(total_cost=0.0, record_count=0, records=[])


def test_non_2xx_raises(monkeypatch):
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(401, text="unauthorized")

    install_transport(monkeypatch, handler)
    with pytest.raises(LangfuseClientError, match="401"):
        make_adapter().fetch_logged_cost(make_window())


def test_runtime_protocol_conformance():
    assert isinstance(make_adapter(), CostLogSource)


def test_request_shape(monkeypatch):
    def handler(request: httpx.Request) -> httpx.Response:
        page = int(request.url.params["page"])
        return httpx.Response(
            200,
            json={
                "data": []
                if page > 1
                else [
                    {"id": "t1", "totalCost": 0.1, "timestamp": "2026-06-03T10:01:00Z"},
                ]
            },
        )

    captured = install_transport(monkeypatch, handler)
    make_adapter().fetch_logged_cost(make_window())

    first = captured[0]
    assert first.url.path == "/api/public/traces"
    assert first.url.params["fromTimestamp"] == "2026-06-03T10:00:00+00:00"
    assert first.url.params["toTimestamp"] == "2026-06-03T11:00:00+00:00"
    expected_auth = "Basic " + base64.b64encode(b"pub:sec").decode()
    assert first.headers["Authorization"] == expected_auth


def test_meta_total_pages_stops_pagination(monkeypatch):
    """When the response carries ``meta.totalPages``, paging stops at that page
    even if the last page is non-empty."""
    data = [{"id": "t1", "totalCost": 1.0, "timestamp": "2026-06-03T10:01:00Z"}]

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"data": data, "meta": {"totalPages": 1}})

    captured = install_transport(monkeypatch, handler)
    result = make_adapter().fetch_logged_cost(make_window())

    assert result.record_count == 1
    assert len(captured) == 1

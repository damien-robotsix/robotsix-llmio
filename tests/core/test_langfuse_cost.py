"""Unit tests for ``LangfuseCostLogSource`` and its helpers.

Drives the Langfuse REST adapter offline via ``httpx.MockTransport``, covering
``fetch_logged_cost``, ``fetch_logged_cost_by_provider``, ``prune_before``,
timestamp parsing, and the observation provider/cost extraction helpers.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime

import httpx
import pytest
from conftest import install_transport, make_adapter, make_window

from robotsix_llmio.core.langfuse_client import LangfuseClientError
from robotsix_llmio.core.langfuse_cost import (
    _observation_cost,
    _observation_provider,
    _parse_timestamp,
)


# --------------------------------------------------------------------------- #
# _parse_timestamp
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    ["input", "expected"],
    [
        ("2024-01-01T12:00:00Z", datetime(2024, 1, 1, 12, 0, tzinfo=UTC)),
        ("2024-01-01T12:00:00+00:00", datetime(2024, 1, 1, 12, 0, tzinfo=UTC)),
        pytest.param(
            datetime(2024, 1, 1, 12, 0, tzinfo=UTC),
            None,
            id="datetime_passthrough",
        ),
        pytest.param("not-a-timestamp", None, id="invalid_raises"),
    ],
)
def test_parse_timestamp(input, expected):
    """``_parse_timestamp`` handles Z/offset suffixes, datetime passthrough,
    and raises ``ValueError`` for unparseable strings."""
    if expected is None and not isinstance(input, str):
        assert _parse_timestamp(input) is input
    elif expected is None:
        with pytest.raises(ValueError):
            _parse_timestamp(input)
    else:
        assert _parse_timestamp(input) == expected


# --------------------------------------------------------------------------- #
# _observation_provider
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    ["observation", "expected"],
    [
        ({"metadata": {"provider": "openrouter"}}, "openrouter"),
        ({}, None),
        ({"metadata": {"other": "x"}}, None),
    ],
)
def test_observation_provider(observation, expected):
    assert _observation_provider(observation) == expected


# --------------------------------------------------------------------------- #
# _observation_cost
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    ["observation", "expected"],
    [
        ({"calculatedTotalCost": 1.0, "totalCost": 2.0}, 1.0),
        ({"totalCost": 2.0}, 2.0),
        ({"costDetails": {"total": 3.0}}, 3.0),
        ({}, 0.0),
        ({"costDetails": {}}, 0.0),
    ],
)
def test_observation_cost(observation, expected):
    assert _observation_cost(observation) == pytest.approx(expected)


# --------------------------------------------------------------------------- #
# fetch_logged_cost
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    ["pages", "expected_count", "expected_total"],
    [
        pytest.param(
            [
                [
                    {"id": "t1", "totalCost": 0.5, "timestamp": "2026-06-03T10:01:00Z"},
                    {"id": "t2", "totalCost": 1.5, "timestamp": "2026-06-03T10:02:00Z"},
                ],
                [],
            ],
            2,
            2.0,
            id="single_page",
        ),
        pytest.param(
            [
                [{"id": "t1", "totalCost": 1.0, "timestamp": "2026-06-03T10:01:00Z"}],
                [{"id": "t2", "totalCost": 2.0, "timestamp": "2026-06-03T10:02:00Z"}],
                [],
            ],
            2,
            3.0,
            id="multi_page_break",
        ),
        pytest.param(
            [
                [
                    {
                        "id": "t1",
                        "totalCost": 0.75,
                        "timestamp": "2026-06-03T10:01:00Z",
                        "sessionId": "sess-1",
                        "name": "trace-one",
                    },
                    {
                        "id": "t2",
                        "totalCost": 0.25,
                        "timestamp": "2026-06-03T10:02:00Z",
                        "sessionId": "sess-2",
                        "name": "trace-two",
                    },
                ],
                [],
            ],
            2,
            1.0,
            id="record_aggregation",
        ),
        pytest.param(
            [[]],
            0,
            0.0,
            id="empty_data",
        ),
    ],
)
def test_fetch_logged_cost(monkeypatch, pages, expected_count, expected_total):
    """``fetch_logged_cost`` paginates traces, aggregates costs, and maps
    ``CostRecord`` fields correctly."""
    page_iter = iter(pages)

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"data": next(page_iter)})

    install_transport(monkeypatch, handler)
    result = make_adapter().fetch_logged_cost(make_window())

    assert result.record_count == expected_count
    assert result.total_cost == pytest.approx(expected_total)

    # Verify CostRecord field mapping against the input data.
    all_input = [r for page in pages for r in page]
    for src, record in zip(all_input, result.records, strict=False):
        assert record.id == src["id"]
        assert record.cost == pytest.approx(src.get("totalCost", 0))
        if "sessionId" in src:
            assert record.session_id == src["sessionId"]
        if "name" in src:
            assert record.name == src["name"]
        assert isinstance(record.timestamp, datetime)


def test_fetch_logged_cost_http_error(monkeypatch):
    """A non-2xx response raises ``LangfuseClientError``."""

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(500, text="boom")

    install_transport(monkeypatch, handler)
    with pytest.raises(LangfuseClientError, match="500"):
        make_adapter().fetch_logged_cost(make_window())


# --------------------------------------------------------------------------- #
# fetch_logged_cost_by_provider
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    ["pages", "provider", "expected_count", "expected_total"],
    [
        pytest.param(
            [
                [
                    {
                        "id": "o1",
                        "calculatedTotalCost": 0.4,
                        "startTime": "2026-06-03T10:01:00Z",
                        "traceId": "tr1",
                        "name": "gen-1",
                        "metadata": {"provider": "openrouter"},
                    },
                    {
                        "id": "o2",
                        "calculatedTotalCost": 9.9,
                        "startTime": "2026-06-03T10:02:00Z",
                        "metadata": {"provider": "claude-sdk"},
                    },
                ],
                [
                    {
                        "id": "o3",
                        "calculatedTotalCost": 0.6,
                        "startTime": "2026-06-03T10:03:00Z",
                        "traceId": "tr3",
                        "metadata": {"provider": "openrouter"},
                    },
                ],
                [],
            ],
            "openrouter",
            2,
            1.0,
            id="filters_and_paginates",
        ),
        pytest.param(
            [
                [
                    {
                        "id": "calc",
                        "calculatedTotalCost": 1.0,
                        "totalCost": 99.0,
                        "startTime": "2026-06-03T10:01:00Z",
                        "metadata": {"provider": "p"},
                    },
                    {
                        "id": "total",
                        "totalCost": 2.0,
                        "startTime": "2026-06-03T10:02:00Z",
                        "metadata": {"provider": "p"},
                    },
                    {
                        "id": "details",
                        "costDetails": {"total": 3.0},
                        "startTime": "2026-06-03T10:03:00Z",
                        "metadata": {"provider": "p"},
                    },
                ],
                [],
            ],
            "p",
            3,
            6.0,
            id="cost_extraction_order",
        ),
    ],
)
def test_fetch_by_provider(
    monkeypatch, pages, provider, expected_count, expected_total
):
    """``fetch_logged_cost_by_provider`` paginates observations, filters by
    provider, and extracts cost in the correct priority order."""
    page_iter = iter(pages)

    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/api/public/observations"
        assert request.url.params["type"] == "GENERATION"
        return httpx.Response(200, json={"data": next(page_iter)})

    install_transport(monkeypatch, handler)
    result = make_adapter().fetch_logged_cost_by_provider(make_window(), provider)

    assert result.record_count == expected_count
    assert result.total_cost == pytest.approx(expected_total)

    # Verify field mapping for matching records.
    all_input = [
        r
        for page in pages
        for r in page
        if r.get("metadata", {}).get("provider") == provider
    ]
    for src, record in zip(all_input, result.records, strict=False):
        assert record.id == src["id"]
        if "traceId" in src:
            assert record.session_id == src["traceId"]
        if "name" in src:
            assert record.name == src["name"]


def test_fetch_by_provider_http_error(monkeypatch):
    """A non-2xx response raises ``LangfuseClientError``."""

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(403, text="forbidden")

    install_transport(monkeypatch, handler)
    with pytest.raises(LangfuseClientError, match="403"):
        make_adapter().fetch_logged_cost_by_provider(make_window(), "openrouter")


# --------------------------------------------------------------------------- #
# prune_before
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    ["get_responses", "expected_count", "expected_delete_bodies", "expected_get_count"],
    [
        pytest.param(
            [[{"id": "t1"}, {"id": "t2"}], [{"id": "t3"}], []],
            3,
            [["t1", "t2"], ["t3"]],
            None,
            id="deletes_and_counts",
        ),
        pytest.param(
            [[]],
            0,
            [],
            None,
            id="empty_returns_zero",
        ),
        pytest.param(
            [[{"id": "t1"}, {"id": "t2"}], [{"id": "t1"}, {"id": "t2"}]],
            2,
            [["t1", "t2"]],
            2,
            id="delayed_deletion_terminates",
        ),
    ],
)
def test_prune_before(
    monkeypatch,
    get_responses,
    expected_count,
    expected_delete_bodies,
    expected_get_count,
):
    """``prune_before`` deletes traces ≤ cutoff, handles empty pages, and
    terminates on delayed-deletion lag."""
    cutoff = datetime(2026, 6, 1, tzinfo=UTC)
    get_iter = iter(get_responses)
    deleted_bodies: list[list[str]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/api/public/traces"
        if request.method == "GET":
            assert request.url.params["toTimestamp"] == cutoff.isoformat()
            return httpx.Response(200, json={"data": next(get_iter)})
        assert request.method == "DELETE"
        body = json.loads(request.content)
        deleted_bodies.append(body["traceIds"])
        return httpx.Response(200, json={})

    captured = install_transport(monkeypatch, handler)
    count = make_adapter().prune_before(cutoff)

    assert count == expected_count
    assert deleted_bodies == expected_delete_bodies
    if expected_get_count is not None:
        assert sum(1 for r in captured if r.method == "GET") == expected_get_count


@pytest.mark.parametrize(
    "error_on",
    ["get", "delete"],
)
def test_prune_before_http_error(monkeypatch, error_on):
    """A non-2xx response on the GET or DELETE phase raises
    ``LangfuseClientError``."""

    def handler(request: httpx.Request) -> httpx.Response:
        if error_on == "get":
            return httpx.Response(503, text="unavailable")
        # error_on == "delete"
        if request.method == "GET":
            return httpx.Response(200, json={"data": [{"id": "t1"}]})
        return httpx.Response(500, text="delete-failed")

    expected_match = "503" if error_on == "get" else "delete"

    install_transport(monkeypatch, handler)
    with pytest.raises(LangfuseClientError, match=expected_match):
        make_adapter().prune_before(datetime(2026, 6, 1, tzinfo=UTC))


def test_prune_before_max_iterations_raises(monkeypatch):
    """``LangfuseClientError`` is raised when ``_MAX_PRUNE_ITERATIONS`` is
    exceeded."""
    import robotsix_llmio.core.langfuse_cost as _lcm

    monkeypatch.setattr(_lcm, "_MAX_PRUNE_ITERATIONS", 2)

    call_count = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal call_count
        if request.method == "GET":
            call_count += 1
            return httpx.Response(200, json={"data": [{"id": f"t{call_count}"}]})
        return httpx.Response(200, json={})

    install_transport(monkeypatch, handler)
    with pytest.raises(LangfuseClientError, match="iterations"):
        make_adapter().prune_before(datetime(2026, 6, 1, tzinfo=UTC))

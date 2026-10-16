"""``httpx`` mock-response and handler factories for the test suite.

These centralize the ad-hoc ``httpx.Response`` / ``httpx.MockTransport``
handlers that were previously inlined across the Langfuse-cost, knowledge,
and refdocs test modules. Pair them with the transport installers in
``tests/core/conftest.py``:

* ``install_transport`` — patches the synchronous ``httpx.Client`` (Langfuse
  cost source).
* ``install_timeout_transport`` — patches a module's ``timeout_http_client``
  (refdocs client).
* ``install_async_transport`` — patches the asynchronous ``httpx.AsyncClient``.

The JSON field names below mirror exactly what the production parsers read
(see ``core/langfuse_cost.py``, ``clients/knowledge``, ``clients/refdocs``),
so faked payloads stay in sync with the code under test.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Sequence
from typing import Any

import httpx

Handler = Callable[[httpx.Request], httpx.Response]


# --------------------------------------------------------------------------- #
# Langfuse traces / observations
# --------------------------------------------------------------------------- #
def make_langfuse_trace(
    *,
    id: str = "trace-1",
    timestamp: str = "2026-06-03T10:30:00Z",
    total_cost: float | None = 0.0,
    session_id: str | None = None,
    name: str | None = None,
) -> dict[str, Any]:
    """Build one Langfuse *trace* record as returned under ``data``.

    Keys match what ``LangfuseCostLogSource._to_record`` reads: ``id``,
    ``timestamp``, ``totalCost``, ``sessionId``, ``name``.
    """
    return {
        "id": id,
        "timestamp": timestamp,
        "totalCost": total_cost,
        "sessionId": session_id,
        "name": name,
    }


def make_langfuse_observation(
    *,
    id: str = "obs-1",
    start_time: str = "2026-06-03T10:30:00Z",
    trace_id: str | None = "trace-1",
    name: str | None = None,
    calculated_total_cost: float | None = None,
    total_cost: float | None = None,
    cost_details: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build one Langfuse *observation* record as returned under ``data``.

    Keys match what ``_observation_to_record`` / ``_observation_cost`` read:
    ``id``, ``startTime``, ``traceId``, ``name``, and the cost fields
    (``calculatedTotalCost`` preferred, then ``totalCost``, then nested
    ``costDetails.total``). Only the cost fields you pass are included.
    """
    record: dict[str, Any] = {
        "id": id,
        "startTime": start_time,
        "traceId": trace_id,
        "name": name,
    }
    if calculated_total_cost is not None:
        record["calculatedTotalCost"] = calculated_total_cost
    if total_cost is not None:
        record["totalCost"] = total_cost
    if cost_details is not None:
        record["costDetails"] = cost_details
    return record


def make_langfuse_page(
    records: Iterable[dict[str, Any]], *, total_pages: int | None = None
) -> dict[str, Any]:
    """Wrap trace/observation records in the paginated list-API envelope.

    Produces ``{"data": [...], "meta": {"totalPages": N}}`` — ``meta`` is
    included only when ``total_pages`` is given (``iter_pages`` uses it for
    early termination).
    """
    body: dict[str, Any] = {"data": list(records)}
    if total_pages is not None:
        body["meta"] = {"totalPages": total_pages}
    return body


def make_langfuse_traces_response(
    records: Iterable[dict[str, Any]],
    *,
    total_pages: int | None = None,
    status: int = 200,
) -> httpx.Response:
    """Build a Langfuse list-API ``httpx.Response`` from trace records."""
    return httpx.Response(
        status, json=make_langfuse_page(records, total_pages=total_pages)
    )


# --------------------------------------------------------------------------- #
# Knowledge client
# --------------------------------------------------------------------------- #
def make_knowledge_result(
    *,
    id: str = "doc-1",
    title: str = "Untitled",
    snippet: str = "",
    score: float = 0.0,
) -> dict[str, Any]:
    """Build one knowledge-client result (``id``/``title``/``snippet``/``score``)."""
    return {"id": id, "title": title, "snippet": snippet, "score": score}


def make_knowledge_client_response(
    results: Iterable[dict[str, Any]] | None = None, *, status: int = 200
) -> httpx.Response:
    """Build a knowledge-client search ``httpx.Response`` (``{"results": [...]}``)."""
    return httpx.Response(status, json={"results": list(results or [])})


# --------------------------------------------------------------------------- #
# Refdocs client
# --------------------------------------------------------------------------- #
def make_refdocs_result(
    *, path: str = "config/index", title: str = "Untitled", **extra: Any
) -> dict[str, Any]:
    """Build one refdocs search result (at least ``path`` and ``title``)."""
    return {"path": path, "title": title, **extra}


def make_refdocs_response(
    results: Iterable[dict[str, Any]] | None = None, *, status: int = 200
) -> httpx.Response:
    """Build a refdocs search ``httpx.Response`` (``{"results": [...]}``)."""
    return httpx.Response(status, json={"results": list(results or [])})


# --------------------------------------------------------------------------- #
# Generic helpers
# --------------------------------------------------------------------------- #
def make_error_response(status: int, message: str = "error") -> httpx.Response:
    """Build an error ``httpx.Response`` (``{"error": message}``) at *status*."""
    return httpx.Response(status, json={"error": message})


def json_handler(response: httpx.Response) -> Handler:
    """Return a ``MockTransport`` handler that always replies with *response*.

    >>> handler = json_handler(make_knowledge_client_response([]))
    >>> capture = install_transport(monkeypatch, handler, module=mod)  # doctest: +SKIP
    """

    def _handler(request: httpx.Request) -> httpx.Response:
        return response

    return _handler


def paged_handler(pages: Sequence[httpx.Response]) -> Handler:
    """Return a handler that replies with each response in *pages* in turn.

    Useful for pagination tests: successive requests receive successive
    pages; the final page is repeated once exhausted.
    """
    remaining = list(pages)

    def _handler(request: httpx.Request) -> httpx.Response:
        if len(remaining) > 1:
            return remaining.pop(0)
        return remaining[0]

    return _handler

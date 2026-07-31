"""Direct-construction unit tests for the relocated _StampProcessor — no
TracerProvider, just a fake span. Gated on the tracing extra."""

from __future__ import annotations

import threading
from unittest.mock import Mock

import pytest

from robotsix_llmio.core import tracing
from robotsix_llmio.core.tracing import (
    LANGFUSE_PUBLIC_KEY,
    LANGFUSE_SESSION_ID,
    LANGFUSE_TRACE_NAME,
)
from tests.core._fake_span import _FakeSpan


def test_stamp_processor_direct_construction_tiers(monkeypatch):
    """Construct _StampProcessor directly (no OTel pipeline) and drive each
    of the three routing tiers plus the on_end cleanup path."""
    pytest.importorskip("opentelemetry.sdk.trace")
    from robotsix_llmio.core._tracing_processors import _StampProcessor

    monkeypatch.setattr(tracing, "_projects", {"pk-x": {"base_url": "u"}})
    monkeypatch.setattr(tracing, "_default_public_key", "pk-x")
    monkeypatch.setattr(tracing, "_trace_routing", {})
    monkeypatch.setattr(tracing, "_trace_named", set())

    proc = _StampProcessor()

    # Tier 1: contextvar set → stamps session + that public key.
    sess_token = tracing._current_session.set("sess-1")
    pk_token = tracing._current_public_key.set("pk-ctx")
    try:
        span1 = _FakeSpan(trace_id=111)
        proc.on_start(span1)
    finally:
        tracing._current_public_key.reset(pk_token)
        tracing._current_session.reset(sess_token)
    assert span1.attributes["session.id"] == "sess-1"
    assert span1.attributes[LANGFUSE_SESSION_ID] == "sess-1"
    assert span1.attributes[LANGFUSE_PUBLIC_KEY] == "pk-ctx"
    assert tracing._trace_routing[111] == "pk-ctx"

    # Tier 2: contextvar None but trace-level routing pre-populated.
    monkeypatch.setattr(tracing, "_trace_routing", {222: "pk-inherited"})
    span2 = _FakeSpan(trace_id=222)
    proc.on_start(span2)
    assert span2.attributes[LANGFUSE_PUBLIC_KEY] == "pk-inherited"
    assert "session.id" not in span2.attributes

    # Tier 3: contextvar None, no trace entry, single-tenant default.
    monkeypatch.setattr(tracing, "_trace_routing", {})
    span3 = _FakeSpan(trace_id=333)
    proc.on_start(span3)
    assert span3.attributes[LANGFUSE_PUBLIC_KEY] == "pk-x"
    assert tracing._trace_routing[333] == "pk-x"

    # on_end on a root span (parent=None) removes the trace_id mapping.
    proc.on_end(span3)
    assert 333 not in tracing._trace_routing


def test_stamp_processor_names_root_trace(monkeypatch):
    """The trace is named once: an explicit root-span name is preferred, else
    the session label; the session fallback never overwrites it."""
    pytest.importorskip("opentelemetry.sdk.trace")
    from robotsix_llmio.core._tracing_processors import _StampProcessor

    monkeypatch.setattr(tracing, "_projects", {"pk-x": {"base_url": "u"}})
    monkeypatch.setattr(tracing, "_default_public_key", "pk-x")
    monkeypatch.setattr(tracing, "_trace_routing", {})
    monkeypatch.setattr(tracing, "_trace_named", set())
    proc = _StampProcessor()

    token = tracing._current_session.set("robotsix-mill · ticket-xyz")
    try:
        # Empty-named root under a session → name falls back to the session.
        root_empty = _FakeSpan(trace_id=1, parent=None, name="")
        proc.on_start(root_empty)
        # A later child in the same trace must NOT overwrite the name.
        child = _FakeSpan(
            trace_id=1, parent=root_empty.get_span_context(), name="chat opus"
        )
        proc.on_start(child)
        # A named root keeps its explicit name (e.g. a mill stage).
        root_named = _FakeSpan(trace_id=2, parent=None, name="implement")
        proc.on_start(root_named)
    finally:
        tracing._current_session.reset(token)

    assert root_empty.attributes[LANGFUSE_TRACE_NAME] == "robotsix-mill · ticket-xyz"
    assert LANGFUSE_TRACE_NAME not in child.attributes  # not re-stamped
    assert root_named.attributes[LANGFUSE_TRACE_NAME] == "implement"


def test_stamp_processor_child_names_trace_when_root_lost_session(monkeypatch):
    """If the root starts without the session contextvar (cross-thread loss) and
    has no name, the first later span carrying the session names the trace."""
    pytest.importorskip("opentelemetry.sdk.trace")
    from robotsix_llmio.core._tracing_processors import _StampProcessor

    monkeypatch.setattr(tracing, "_projects", {"pk-x": {"base_url": "u"}})
    monkeypatch.setattr(tracing, "_default_public_key", "pk-x")
    monkeypatch.setattr(tracing, "_trace_routing", {})
    monkeypatch.setattr(tracing, "_trace_named", set())
    proc = _StampProcessor()

    # Root: no session in context, empty name → unnameable at its on_start.
    root = _FakeSpan(trace_id=7, parent=None, name="")
    proc.on_start(root)
    assert LANGFUSE_TRACE_NAME not in root.attributes

    # A later child runs in a context that carries the session → names trace.
    token = tracing._current_session.set("robotsix-mill · ticket-7")
    try:
        child = _FakeSpan(trace_id=7, parent=root.get_span_context(), name="chat opus")
        proc.on_start(child)
    finally:
        tracing._current_session.reset(token)
    assert child.attributes[LANGFUSE_TRACE_NAME] == "robotsix-mill · ticket-7"

    proc.on_end(root)  # root end clears the per-trace guard
    assert 7 not in tracing._trace_named


def test_stamp_processor_root_unnamed_without_session(monkeypatch):
    """An empty-named root with no session and no later session-bearing span
    gets no trace name — and never crashes."""
    pytest.importorskip("opentelemetry.sdk.trace")
    from robotsix_llmio.core._tracing_processors import _StampProcessor

    monkeypatch.setattr(tracing, "_projects", {"pk-x": {"base_url": "u"}})
    monkeypatch.setattr(tracing, "_default_public_key", "pk-x")
    monkeypatch.setattr(tracing, "_trace_routing", {})
    monkeypatch.setattr(tracing, "_trace_named", set())
    proc = _StampProcessor()

    root = _FakeSpan(trace_id=9, parent=None, name="")
    proc.on_start(root)
    assert LANGFUSE_TRACE_NAME not in root.attributes


def test_stamp_processor_concurrent_trace_name_guard(monkeypatch):
    """Two spans of the same trace starting concurrently must set the trace
    name attribute at most once — the child must never overwrite the root's
    explicit name with the session-id fallback."""
    pytest.importorskip("opentelemetry.sdk.trace")
    from robotsix_llmio.core._tracing_processors import _StampProcessor

    monkeypatch.setattr(tracing, "_projects", {"pk-x": {"base_url": "u"}})
    monkeypatch.setattr(tracing, "_default_public_key", "pk-x")
    monkeypatch.setattr(tracing, "_trace_routing", {})
    monkeypatch.setattr(tracing, "_trace_named", set())
    proc = _StampProcessor()

    # Two spans sharing the same trace_id:
    # - root: root span with explicit name "implement"
    # - child: a later span that carries the session contextvar
    root = _FakeSpan(trace_id=42, parent=None, name="implement")
    child = _FakeSpan(trace_id=42, parent=root.get_span_context(), name="chat opus")

    # Replace set_attribute with a mock so we can assert call count.
    root_setattr = Mock(wraps=root.set_attribute)
    root.set_attribute = root_setattr
    child_setattr = Mock(wraps=child.set_attribute)
    child.set_attribute = child_setattr

    errors = []
    token = tracing._current_session.set("robotsix-mill · ticket-concurrent")

    def start_root() -> None:
        try:
            proc.on_start(root)
        except Exception as exc:
            errors.append(exc)

    def start_child() -> None:
        try:
            proc.on_start(child)
        except Exception as exc:
            errors.append(exc)

    try:
        t1 = threading.Thread(target=start_root)
        t2 = threading.Thread(target=start_child)
        t1.start()
        t2.start()
        t1.join()
        t2.join()
    finally:
        tracing._current_session.reset(token)

    assert not errors, f"on_start raised: {errors}"

    # Exactly one span (the root) should set the trace name.
    root_calls = root_setattr.call_args_list
    child_calls = child_setattr.call_args_list
    root_name_calls = [
        c for c in root_calls if c.args and c.args[0] == LANGFUSE_TRACE_NAME
    ]
    child_name_calls = [
        c for c in child_calls if c.args and c.args[0] == LANGFUSE_TRACE_NAME
    ]

    assert len(root_name_calls) == 1, (
        f"root should get trace name exactly once, got {len(root_name_calls)}"
    )
    assert len(child_name_calls) == 0, (
        f"child must not set trace name, got {len(child_name_calls)}"
    )
    assert root_name_calls[0].args[1] == "implement"

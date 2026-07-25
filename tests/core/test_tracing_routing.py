"""Multi-tenant trace routing + span processor unit tests.

Covers trace-level routing (Tier 1/2/3), the relocated _StampProcessor and
_FilteredBatchSpanProcessor, all exercised without a global TracerProvider install.
"""

from __future__ import annotations

import contextlib
import threading
from unittest.mock import Mock

import pytest

from robotsix_llmio.core import tracing
from robotsix_llmio.core.tracing import (
    LANGFUSE_PUBLIC_KEY,
    LANGFUSE_SESSION_ID,
    LANGFUSE_TRACE_NAME,
)

# ---------------------------------------------------------------------------
# Trace-level routing tests (multi-tenant span processor)
# ---------------------------------------------------------------------------


@contextlib.contextmanager
def _setup_tracing_pipeline(monkeypatch):
    """Set up a real TracerProvider with _StampProcessor + InMemorySpanExporter.

    Mocks OTLPSpanExporter and Agent.instrument_all so no network calls occur.
    Resets module-level state so each test starts clean. Returns the
    InMemorySpanExporter for reading back finished spans.
    """
    pytest.importorskip("opentelemetry.exporter.otlp.proto.http.trace_exporter")
    pytest.importorskip("pydantic_ai")

    from opentelemetry import trace as otel_trace
    from opentelemetry.exporter.otlp.proto.http import trace_exporter as _te
    from opentelemetry.sdk.trace.export import (
        SimpleSpanProcessor,
        SpanExportResult,
    )
    from opentelemetry.sdk.trace.export.in_memory_span_exporter import (
        InMemorySpanExporter,
    )

    # Reset module-level globals so each test starts from a known state.
    monkeypatch.setattr(tracing, "_provider", None)
    monkeypatch.setattr(tracing, "_projects", {})
    monkeypatch.setattr(tracing, "_default_public_key", None)
    monkeypatch.setattr(tracing, "_trace_routing", {})
    monkeypatch.setattr(tracing, "_trace_named", set())
    monkeypatch.setattr(tracing, "_warn_last_ts", 0.0)
    monkeypatch.setattr(tracing, "_debug_last_ts", 0.0)

    # Mock OTLPSpanExporter — no network.
    class FakeExporter:
        def __init__(self, *args, **kwargs):
            pass

        def export(self, spans):
            return SpanExportResult.SUCCESS

        def shutdown(self):
            pass

    monkeypatch.setattr(_te, "OTLPSpanExporter", FakeExporter)

    # Mock Agent.instrument_all — avoid pydantic-ai side effects.
    import pydantic_ai

    monkeypatch.setattr(pydantic_ai.Agent, "instrument_all", lambda: None)

    # Save / restore the global OTel tracer provider so other tests aren't
    # contaminated.
    old_provider = otel_trace.get_tracer_provider()

    assert (
        tracing.setup_langfuse_tracing(public_key="pk-test", secret_key="sk-test")
        is True
    )

    mem = InMemorySpanExporter()
    tracing._provider.add_span_processor(SimpleSpanProcessor(mem))

    yield mem

    # Clean up.
    tracing._provider.shutdown()
    tracing._provider = None
    otel_trace.set_tracer_provider(old_provider)


def _get_span_by_name(mem, name: str):
    """Return the first finished span named *name*, or None."""
    for span in mem.get_finished_spans():
        if span.name == name:
            return span
    return None


def test_trace_routing_inheritance_from_root(monkeypatch):
    """A child span inherits the routing key from the root span's trace
    mapping even when the contextvar is cleared."""
    with _setup_tracing_pipeline(monkeypatch) as mem:
        from opentelemetry import trace

        tracer = trace.get_tracer("test")

        with tracing.langfuse_project("pk-root"), tracer.start_as_current_span("root"):
            # Clear the contextvar inside the root span (simulating a
            # thread-pool task that lost it), then create a child.
            token = tracing._current_public_key.set(None)
            try:
                with tracer.start_as_current_span("child"):
                    pass
            finally:
                tracing._current_public_key.reset(token)

        root_span = _get_span_by_name(mem, "root")
        child_span = _get_span_by_name(mem, "child")
        assert root_span is not None
        assert child_span is not None

        # Root got pk-root from the contextvar.
        assert root_span.attributes.get(LANGFUSE_PUBLIC_KEY) == "pk-root"
        # Child inherited pk-root from the trace-level mapping (Tier 2).
        assert child_span.attributes.get(LANGFUSE_PUBLIC_KEY) == "pk-root"


def test_trace_routing_no_fallback_multi_tenant(monkeypatch):
    """With ≥2 registered projects and no contextvar, no routing key is
    stamped (span is unroutable)."""
    with _setup_tracing_pipeline(monkeypatch) as mem:
        from opentelemetry import trace

        # Register a second project.
        assert (
            tracing.setup_langfuse_tracing(public_key="pk-extra", secret_key="sk-extra")
            is True
        )

        tracer = trace.get_tracer("test")
        with tracer.start_as_current_span("orphan"):
            pass

        span = _get_span_by_name(mem, "orphan")
        assert span is not None
        assert LANGFUSE_PUBLIC_KEY not in (span.attributes or {})


def test_trace_routing_fallback_single_tenant(monkeypatch):
    """With exactly one registered project and no contextvar, the span
    receives the default_public_key (backward compat)."""
    with _setup_tracing_pipeline(monkeypatch) as mem:
        from opentelemetry import trace

        # Only one project registered (pk-test), no contextvar active.
        tracer = trace.get_tracer("test")
        with tracer.start_as_current_span("single"):
            pass

        span = _get_span_by_name(mem, "single")
        assert span is not None
        assert span.attributes.get(LANGFUSE_PUBLIC_KEY) == "pk-test"


def test_trace_routing_root_cleanup(monkeypatch):
    """When a root span ends, its trace_id is removed from _trace_routing."""
    with _setup_tracing_pipeline(monkeypatch):
        from opentelemetry import trace

        tracer = trace.get_tracer("test")

        with tracing.langfuse_project("pk-cleanup"):
            with tracer.start_as_current_span("root") as root:
                trace_id = root.get_span_context().trace_id
                # The trace_id must have been recorded while the span is alive.
                with tracing._trace_routing_lock:
                    assert tracing._trace_routing.get(trace_id) == "pk-cleanup"

            # After the root span ends, the entry must be removed.
            with tracing._trace_routing_lock:
                assert trace_id not in tracing._trace_routing


def test_unrouted_span_warning(monkeypatch, caplog):
    """In multi-tenant mode, an unroutable span emits a WARNING log."""
    with _setup_tracing_pipeline(monkeypatch):
        from opentelemetry import trace

        # Register a second project to enter multi-tenant mode.
        tracing.setup_langfuse_tracing(public_key="pk-extra", secret_key="sk-extra")

        tracer = trace.get_tracer("test")
        with tracer.start_as_current_span("orphan"):
            pass

        warnings = [r for r in caplog.records if r.levelname == "WARNING"]
        assert any(
            "Cannot route span" in r.message and "multi-tenant" in r.message
            for r in warnings
        ), (
            f"expected throttled multi-tenant warning, "
            f"got: {[r.message for r in warnings]}"
        )


def test_active_routing_key():
    """active_routing_key() returns the contextvar value, never the default."""
    assert tracing.active_routing_key() is None

    with tracing.langfuse_project("pk-x"):
        assert tracing.active_routing_key() == "pk-x"

    assert tracing.active_routing_key() is None


# ---------------------------------------------------------------------------
# Direct-construction unit test for the relocated _StampProcessor — no
# TracerProvider, just a fake span. Gated on the tracing extra.
# ---------------------------------------------------------------------------


class _FakeSpanContext:
    def __init__(self, trace_id: int) -> None:
        self.trace_id = trace_id


class _FakeTraceFlags:
    sampled = True


class _FakeContext:
    trace_flags = _FakeTraceFlags()


class _FakeSpan:
    """Minimal span exposing what _StampProcessor reads/writes."""

    def __init__(self, trace_id: int, parent=None, name: str = "fake-span") -> None:
        self.attributes: dict = {}
        self._ctx = _FakeSpanContext(trace_id)
        self.parent = parent
        self.name = name
        self.context = _FakeContext()

    def set_attribute(self, key, value):
        self.attributes[key] = value

    def get_span_context(self):
        return self._ctx


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


# --- _FilteredBatchSpanProcessor unit tests --------------------------------


class _ListSpanExporter:
    """Fake exporter that records spans passed to ``export()``."""

    def __init__(self):
        self.exported: list = []

    def export(self, spans):
        self.exported.extend(spans)
        from opentelemetry.sdk.trace.export import SpanExportResult

        return SpanExportResult.SUCCESS

    def shutdown(self):
        pass


def test_filtered_batch_processor_matching_key_passes_through(monkeypatch):
    """Span with a matching ``langfuse.public_key`` reaches the exporter."""
    pytest.importorskip("opentelemetry.sdk.trace")
    from robotsix_llmio.core._tracing_processors import _FilteredBatchSpanProcessor

    exporter = _ListSpanExporter()
    proc = _FilteredBatchSpanProcessor(exporter, target_public_key="pk-a")
    try:
        span = _FakeSpan(trace_id=1)
        span.attributes[LANGFUSE_PUBLIC_KEY] = "pk-a"
        proc.on_end(span)
        proc.force_flush()

        assert len(exporter.exported) == 1
        assert exporter.exported[0] is span
    finally:
        proc.shutdown()


def test_filtered_batch_processor_mismatched_key_is_dropped(monkeypatch):
    """Span with a different ``langfuse.public_key`` is silently dropped."""
    pytest.importorskip("opentelemetry.sdk.trace")
    from robotsix_llmio.core._tracing_processors import _FilteredBatchSpanProcessor

    exporter = _ListSpanExporter()
    proc = _FilteredBatchSpanProcessor(exporter, target_public_key="pk-a")
    try:
        span = _FakeSpan(trace_id=2)
        span.attributes[LANGFUSE_PUBLIC_KEY] = "pk-b"
        proc.on_end(span)
        proc.force_flush()

        assert len(exporter.exported) == 0
    finally:
        proc.shutdown()


def test_filtered_batch_processor_missing_key_logs_and_drops(monkeypatch):
    """Span with no ``langfuse.public_key`` logs throttled-debug and is dropped."""
    pytest.importorskip("opentelemetry.sdk.trace")
    from robotsix_llmio.core._tracing_processors import _FilteredBatchSpanProcessor

    debug_calls = []
    monkeypatch.setattr(tracing, "_throttled_debug", debug_calls.append)

    exporter = _ListSpanExporter()
    proc = _FilteredBatchSpanProcessor(exporter, target_public_key="pk-a")
    try:
        span = _FakeSpan(trace_id=3, name="my-span")
        # No langfuse.public_key set — simulate a span that was never stamped.
        proc.on_end(span)
        proc.force_flush()

        assert len(exporter.exported) == 0
        assert len(debug_calls) == 1
        assert "my-span" in debug_calls[0]
        assert LANGFUSE_PUBLIC_KEY in debug_calls[0]
    finally:
        proc.shutdown()

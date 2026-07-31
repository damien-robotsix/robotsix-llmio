"""Multi-tenant trace routing + span processor unit tests.

Covers trace-level routing (Tier 1/2/3), the relocated _StampProcessor and
_FilteredBatchSpanProcessor, all exercised without a global TracerProvider install.
"""

from __future__ import annotations

import contextlib

import pytest

from robotsix_llmio.core import tracing
from robotsix_llmio.core.tracing import LANGFUSE_PUBLIC_KEY

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

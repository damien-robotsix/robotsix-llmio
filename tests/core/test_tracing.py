"""Langfuse trace-export setup — offline unit tests.

Covers the pure helpers, the credentials-absent no-op, session/project context
vars, the root-span handle, and the flush no-op. The full exporter wiring +
Langfuse round-trip (single- and multi-tenant) is exercised in
``tests/test_tracing_live.py`` (on-demand, gated by ``live``), so the offline
suite never installs a global TracerProvider.
"""

from __future__ import annotations

import base64

import pytest

from robotsix_llmio.core import tracing
from robotsix_llmio.core.tracing import (
    _active_public_key,
    _basic_auth_header,
    _langfuse_otlp_endpoint,
    current_session,
    flush_tracing,
    install_signal_handlers,
    langfuse_project,
    langfuse_session,
    langfuse_trace_url,
    make_session_id,
    setup_langfuse_tracing,
    start_trace,
)

# --- public re-export contract ---------------------------------------------

# Semconv constants consumed by sibling packages (claude_sdk, openrouter) must
# stay importable from the public ``core.tracing`` module and resolve to the
# identical values defined in the private ``core._otel`` single source of truth.
_REEXPORTED_SEMCONV_NAMES = (
    "OP_CHAT",
    "OP_EXECUTE_TOOL",
    "OP_INVOKE_AGENT",
    "GEN_AI_OPERATION_NAME",
    "GEN_AI_PROVIDER_NAME",
    "GEN_AI_REQUEST_MODEL",
    "GEN_AI_SYSTEM",
    "GEN_AI_TOOL_NAME",
    "GEN_AI_USAGE_COST",
    "GEN_AI_USAGE_INPUT_TOKENS",
    "GEN_AI_USAGE_OUTPUT_TOKENS",
    "GEN_AI_USAGE_CACHE_READ_INPUT_TOKENS",
    "GEN_AI_USAGE_CACHE_CREATION_INPUT_TOKENS",
    "GEN_AI_USAGE_REASONING_TOKENS",
    "LANGFUSE_OBSERVATION_INPUT",
    "LANGFUSE_OBSERVATION_OUTPUT",
    "LANGFUSE_OBSERVATION_COST_DETAILS",
    "LANGFUSE_OBSERVATION_METADATA_PROVIDER",
    "LANGFUSE_COST_DETAILS_TOTAL_KEY",
)


@pytest.mark.parametrize("name", _REEXPORTED_SEMCONV_NAMES)
def test_semconv_constants_reexported_from_tracing(name):
    from robotsix_llmio.core import _otel

    assert hasattr(tracing, name), f"{name} not re-exported from core.tracing"
    assert getattr(tracing, name) == getattr(_otel, name)


def test_otlp_endpoint_path():
    assert (
        _langfuse_otlp_endpoint("https://cloud.langfuse.com")
        == "https://cloud.langfuse.com/api/public/otel/v1/traces"
    )
    # trailing slash tolerated
    assert (
        _langfuse_otlp_endpoint("https://lf.example.com/")
        == "https://lf.example.com/api/public/otel/v1/traces"
    )


def test_basic_auth_header_is_base64_public_secret():
    header = _basic_auth_header("pk-test", "sk-test")
    assert header.startswith("Basic ")
    assert base64.b64decode(header.split(" ", 1)[1]).decode() == "pk-test:sk-test"


def test_setup_is_noop_without_credentials(monkeypatch):
    monkeypatch.delenv("LANGFUSE_PUBLIC_KEY", raising=False)
    monkeypatch.delenv("LANGFUSE_SECRET_KEY", raising=False)
    assert setup_langfuse_tracing() is False


def test_setup_is_noop_with_only_one_key(monkeypatch):
    monkeypatch.delenv("LANGFUSE_SECRET_KEY", raising=False)
    assert setup_langfuse_tracing(public_key="pk-only") is False


# --- session + project routing context vars --------------------------------


def test_langfuse_session_sets_and_resets_contextvar():
    assert tracing._current_session.get() is None
    with langfuse_session("sess-1"):
        assert tracing._current_session.get() == "sess-1"
        with langfuse_session("sess-2"):  # nesting restores the outer value
            assert tracing._current_session.get() == "sess-2"
        assert tracing._current_session.get() == "sess-1"
    assert tracing._current_session.get() is None


def test_active_public_key_default_and_override(monkeypatch):
    # Default route is the first registered project; langfuse_project overrides.
    monkeypatch.setattr(tracing, "_default_public_key", "pk-default")
    assert _active_public_key() == "pk-default"
    with langfuse_project("pk-other"):
        assert _active_public_key() == "pk-other"
        with langfuse_project("pk-third"):
            assert _active_public_key() == "pk-third"
        assert _active_public_key() == "pk-other"
    assert _active_public_key() == "pk-default"


def test_active_public_key_none_when_no_default(monkeypatch):
    monkeypatch.setattr(tracing, "_default_public_key", None)
    assert _active_public_key() is None


def test_current_session_and_make_session_id():
    assert current_session() is None
    with langfuse_session("s-1"):
        assert current_session() == "s-1"
    sid = make_session_id("review")
    assert sid.startswith("review-") and len(sid) > len("review-")


# --- root-span handle + flush ----------------------------------------------


def test_start_trace_safe_without_provider():
    # No SDK provider in the offline suite → non-recording span → no-op handle.
    with start_trace("offline-trace", session_id="s", project="pk-x") as span:
        span.set_input({"a": 1})  # must not raise
        span.set_output("done")
        assert span.trace_id is None or isinstance(span.trace_id, str)


def test_flush_is_safe_noop_without_provider():
    flush_tracing()


# --- trace URL + signal handlers -------------------------------------------


def test_langfuse_trace_url_builds_from_registered_project(monkeypatch):
    monkeypatch.setattr(
        tracing,
        "_projects",
        {"pk-a": {"base_url": "https://lf.example.com", "project_id": "proj-123"}},
    )
    monkeypatch.setattr(tracing, "_default_public_key", "pk-a")
    # default project
    assert (
        langfuse_trace_url("abc123")
        == "https://lf.example.com/project/proj-123/traces/abc123"
    )
    # explicit project
    assert langfuse_trace_url("abc123", public_key="pk-a").endswith(
        "/project/proj-123/traces/abc123"
    )
    # unknown project -> None
    assert langfuse_trace_url("abc123", public_key="pk-missing") is None


def test_langfuse_trace_url_none_without_project_id(monkeypatch):
    monkeypatch.setattr(
        tracing,
        "_projects",
        {"pk-a": {"base_url": "https://x", "project_id": None}},
    )
    monkeypatch.setattr(tracing, "_default_public_key", "pk-a")
    assert langfuse_trace_url("abc") is None


def test_install_signal_handlers_is_safe():
    import signal

    orig_term = signal.getsignal(signal.SIGTERM)
    orig_int = signal.getsignal(signal.SIGINT)
    try:
        install_signal_handlers()  # must not raise; registers flush-on-signal
    finally:  # restore so we don't affect the rest of the test session
        signal.signal(signal.SIGTERM, orig_term)
        signal.signal(signal.SIGINT, orig_int)


def test_on_export_result_hook_reports_outcomes(monkeypatch):
    """When ``on_export_result`` is supplied, the per-project exporter is
    wrapped so every export attempt reports ``(public_key, ok, error)`` — True
    on success, False (with a message) on a FAILURE result or an exception.

    Isolated from global OTel state: we pre-seed ``tracing._provider`` with a
    throwaway ``TracerProvider`` so ``setup`` skips its one-time global install
    (no ``set_tracer_provider`` / ``instrument_all``) and just wires the
    filtered exporter onto our local provider.

    Needs the ``tracing`` extra (OTLP exporter + SDK); skips without it, the
    same way the rest of the offline suite avoids hard-depending on it.
    """
    pytest.importorskip("opentelemetry.exporter.otlp.proto.http.trace_exporter")
    from opentelemetry.exporter.otlp.proto.http import trace_exporter as _te
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import SpanExportResult

    fresh = TracerProvider()
    monkeypatch.setattr(tracing, "_provider", fresh)
    monkeypatch.setattr(tracing, "_projects", {})

    behavior = {"mode": "success"}

    def fake_export(self, spans):  # no network — controlled outcome
        if behavior["mode"] == "raise":
            raise RuntimeError("boom")
        return (
            SpanExportResult.SUCCESS
            if behavior["mode"] == "success"
            else SpanExportResult.FAILURE
        )

    monkeypatch.setattr(_te.OTLPSpanExporter, "export", fake_export)

    events: list[tuple] = []
    assert (
        setup_langfuse_tracing(
            public_key="pk-hook",
            secret_key="sk-hook",
            base_url="https://lf.example.com",
            on_export_result=lambda pk, ok, err: events.append((pk, ok, err)),
        )
        is True
    )

    # Pull the wrapping exporter back off the provider's filtered processor.
    procs = fresh._active_span_processor._span_processors
    reporting = next(
        p.span_exporter
        for p in procs
        if hasattr(getattr(p, "span_exporter", None), "_hook")
    )

    behavior["mode"] = "success"
    reporting.export([])
    behavior["mode"] = "failure"
    reporting.export([])
    behavior["mode"] = "raise"
    reporting.export([])

    assert events[0] == ("pk-hook", True, None)
    assert events[1][:2] == ("pk-hook", False) and events[1][2]
    assert events[2][:2] == ("pk-hook", False) and "RuntimeError" in events[2][2]


def test_on_export_result_hook_exceptions_never_break_export(monkeypatch):
    """A raising health hook must not propagate out of ``export``."""
    pytest.importorskip("opentelemetry.exporter.otlp.proto.http.trace_exporter")
    from opentelemetry.exporter.otlp.proto.http import trace_exporter as _te
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import SpanExportResult

    fresh = TracerProvider()
    monkeypatch.setattr(tracing, "_provider", fresh)
    monkeypatch.setattr(tracing, "_projects", {})
    monkeypatch.setattr(
        _te.OTLPSpanExporter, "export", lambda self, spans: SpanExportResult.SUCCESS
    )

    def _boom(pk, ok, err):
        raise ValueError("hook blew up")

    assert (
        setup_langfuse_tracing(
            public_key="pk-boom", secret_key="sk-boom", on_export_result=_boom
        )
        is True
    )
    procs = fresh._active_span_processor._span_processors
    reporting = next(
        p.span_exporter
        for p in procs
        if hasattr(getattr(p, "span_exporter", None), "_hook")
    )
    # Must return the underlying result, swallowing the hook's exception.
    assert reporting.export([]) == SpanExportResult.SUCCESS


# ---------------------------------------------------------------------------
# Trace-level routing tests (multi-tenant span processor)
# ---------------------------------------------------------------------------


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
    mem = next(_setup_tracing_pipeline(monkeypatch))
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
    assert root_span.attributes.get("langfuse.public_key") == "pk-root"
    # Child inherited pk-root from the trace-level mapping (Tier 2).
    assert child_span.attributes.get("langfuse.public_key") == "pk-root"


def test_trace_routing_no_fallback_multi_tenant(monkeypatch):
    """With ≥2 registered projects and no contextvar, no routing key is
    stamped (span is unroutable)."""
    mem = next(_setup_tracing_pipeline(monkeypatch))
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
    assert "langfuse.public_key" not in (span.attributes or {})


def test_trace_routing_fallback_single_tenant(monkeypatch):
    """With exactly one registered project and no contextvar, the span
    receives the default_public_key (backward compat)."""
    mem = next(_setup_tracing_pipeline(monkeypatch))
    from opentelemetry import trace

    # Only one project registered (pk-test), no contextvar active.
    tracer = trace.get_tracer("test")
    with tracer.start_as_current_span("single"):
        pass

    span = _get_span_by_name(mem, "single")
    assert span is not None
    assert span.attributes.get("langfuse.public_key") == "pk-test"


def test_trace_routing_root_cleanup(monkeypatch):
    """When a root span ends, its trace_id is removed from _trace_routing."""
    next(_setup_tracing_pipeline(monkeypatch))
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
    next(_setup_tracing_pipeline(monkeypatch))
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
    ), f"expected throttled multi-tenant warning, got: {[r.message for r in warnings]}"


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
    assert span1.attributes["langfuse.session.id"] == "sess-1"
    assert span1.attributes["langfuse.public_key"] == "pk-ctx"
    assert tracing._trace_routing[111] == "pk-ctx"

    # Tier 2: contextvar None but trace-level routing pre-populated.
    monkeypatch.setattr(tracing, "_trace_routing", {222: "pk-inherited"})
    span2 = _FakeSpan(trace_id=222)
    proc.on_start(span2)
    assert span2.attributes["langfuse.public_key"] == "pk-inherited"
    assert "session.id" not in span2.attributes

    # Tier 3: contextvar None, no trace entry, single-tenant default.
    monkeypatch.setattr(tracing, "_trace_routing", {})
    span3 = _FakeSpan(trace_id=333)
    proc.on_start(span3)
    assert span3.attributes["langfuse.public_key"] == "pk-x"
    assert tracing._trace_routing[333] == "pk-x"

    # on_end on a root span (parent=None) removes the trace_id mapping.
    proc.on_end(span3)
    assert 333 not in tracing._trace_routing


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

    span = _FakeSpan(trace_id=1)
    span.attributes["langfuse.public_key"] = "pk-a"
    proc.on_end(span)
    proc.force_flush()

    assert len(exporter.exported) == 1
    assert exporter.exported[0] is span


def test_filtered_batch_processor_mismatched_key_is_dropped(monkeypatch):
    """Span with a different ``langfuse.public_key`` is silently dropped."""
    pytest.importorskip("opentelemetry.sdk.trace")
    from robotsix_llmio.core._tracing_processors import _FilteredBatchSpanProcessor

    exporter = _ListSpanExporter()
    proc = _FilteredBatchSpanProcessor(exporter, target_public_key="pk-a")

    span = _FakeSpan(trace_id=2)
    span.attributes["langfuse.public_key"] = "pk-b"
    proc.on_end(span)
    proc.force_flush()

    assert len(exporter.exported) == 0


def test_filtered_batch_processor_missing_key_logs_and_drops(monkeypatch):
    """Span with no ``langfuse.public_key`` logs throttled-debug and is dropped."""
    pytest.importorskip("opentelemetry.sdk.trace")
    from robotsix_llmio.core._tracing_processors import _FilteredBatchSpanProcessor

    debug_calls = []
    monkeypatch.setattr(tracing, "_throttled_debug", debug_calls.append)

    exporter = _ListSpanExporter()
    proc = _FilteredBatchSpanProcessor(exporter, target_public_key="pk-a")

    span = _FakeSpan(trace_id=3, name="my-span")
    # No langfuse.public_key set — simulate a span that was never stamped.
    proc.on_end(span)
    proc.force_flush()

    assert len(exporter.exported) == 0
    assert len(debug_calls) == 1
    assert "my-span" in debug_calls[0]
    assert "langfuse.public_key" in debug_calls[0]

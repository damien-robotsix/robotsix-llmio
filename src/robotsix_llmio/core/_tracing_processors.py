"""Relocated OpenTelemetry span processors / exporter for Langfuse export.

These three classes used to be defined as function-locals inside
:func:`robotsix_llmio.core.tracing.setup_langfuse_tracing`. They subclass
OpenTelemetry SDK base classes, so they live in this private submodule that is
imported **lazily** (only from within ``setup_langfuse_tracing`` or a test gated
on the ``tracing`` extra) — keeping ``import robotsix_llmio`` working when the
optional ``tracing`` extra is absent.

The classes read ``tracing``'s reassignable module-level globals through the
module object (``_t.*``) so that reassignment in ``setup_langfuse_tracing`` and
``monkeypatch.setattr(tracing, ...)`` in tests are observed live.
"""

from __future__ import annotations

import contextlib

from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
from opentelemetry.sdk.trace import SpanProcessor
from opentelemetry.sdk.trace.export import (
    BatchSpanProcessor,
    SpanExporter,
    SpanExportResult,
)

from . import tracing as _t
from ._otel import LANGFUSE_PUBLIC_KEY, LANGFUSE_SESSION_ID, LANGFUSE_TRACE_NAME


class _StampProcessor(SpanProcessor):
    """Stamp ``session.id`` + target ``langfuse.public_key`` onto every
    span from the active session/project context, so Langfuse groups the
    run and the filtered exporters can route it."""

    def on_start(self, span, parent_context=None):  # type: ignore[no-untyped-def]
        sid = _t._current_session.get()
        if sid:
            span.set_attribute("session.id", sid)
            span.set_attribute(LANGFUSE_SESSION_ID, sid)

        # Ensure the trace root always carries a non-empty Langfuse trace name.
        # Some agent runs reach Langfuse with a pydantic-ai / claude_sdk span as
        # the trace root (callers that don't open an explicitly-named root span),
        # which Langfuse then renders with an empty name ("(unnamed)"). Stamp
        # ``langfuse.trace.name`` on every root span so naming is deterministic
        # regardless of how the caller drove the trace: keep the span's own name
        # when set (preserving an explicit stage/label root), else fall back to
        # the session label. Child spans are untouched, so their observation
        # names are unaffected.
        if span.parent is None:
            trace_name = span.name or sid
            if trace_name:
                span.set_attribute(LANGFUSE_TRACE_NAME, trace_name)

        # Three-tier routing key resolution:
        ctx = span.get_span_context()
        trace_id = ctx.trace_id

        # Tier 1: active contextvar (fast path for synchronous code)
        pk = _t._current_public_key.get()

        # Tier 2: trace-level inheritance (cross-thread / cross-task
        # bridge — survives contextvar loss)
        if pk is None:
            with _t._trace_routing_lock:
                pk = _t._trace_routing.get(trace_id)

        # Tier 3: single-tenant default — only when ≤1 project is
        # registered.  In multi-tenant mode this would route
        # everything to the wrong project, so we skip it.
        if pk is None and len(_t._projects) <= 1:
            pk = _t._default_public_key

        if pk:
            span.set_attribute(LANGFUSE_PUBLIC_KEY, pk)
            # Record so children can inherit via trace-level routing.
            with _t._trace_routing_lock:
                _t._trace_routing[trace_id] = pk
        else:
            _t._throttled_warning(
                "Cannot route span to any Langfuse project: no "
                "contextvar, no trace-level routing, and multi-tenant "
                "mode (≥2 projects). Span will be dropped by all "
                "exporters."
            )

    def on_end(self, span):  # type: ignore[no-untyped-def]
        # When a root span ends, remove its trace-level routing entry
        # so the mapping doesn't grow unbounded.
        if span.parent is None:
            trace_id = span.get_span_context().trace_id
            with _t._trace_routing_lock:
                _t._trace_routing.pop(trace_id, None)

    def shutdown(self):  # type: ignore[no-untyped-def]
        pass

    def force_flush(self, timeout_millis: int = 30000):  # type: ignore[no-untyped-def]
        return True


class _FilteredBatchSpanProcessor(BatchSpanProcessor):
    """Forward a span to this project's exporter only when the span's
    ``langfuse.public_key`` matches — the multi-tenant routing seam."""

    def __init__(self, exporter: SpanExporter, *, target_public_key: str) -> None:
        super().__init__(exporter)
        self._target = target_public_key

    def on_end(self, span):  # type: ignore[no-untyped-def]
        attrs = span.attributes or {}
        if LANGFUSE_PUBLIC_KEY not in attrs:
            _t._throttled_debug(
                f"Span {span.name!r} has no langfuse.public_key "
                f"attribute — will be dropped by all exporters."
            )
            return
        if attrs.get(LANGFUSE_PUBLIC_KEY) != self._target:
            return  # belongs to a different project
        super().on_end(span)


class _ReportingExporter(OTLPSpanExporter):
    """Wrap the OTLP exporter so each export attempt reports its
    outcome to *on_export_result* — letting a consumer surface
    "Langfuse export broken" and auto-clear it on recovery, without
    ever breaking the export path (a raising hook is swallowed)."""

    def __init__(self, *a, _pk: str, _hook, **kw):  # type: ignore[no-untyped-def]
        super().__init__(*a, **kw)
        self._pk = _pk
        self._hook = _hook

    def _report(self, ok: bool, error: str | None) -> None:
        with contextlib.suppress(Exception):
            self._hook(self._pk, ok, error)

    def export(self, spans):  # type: ignore[no-untyped-def]
        try:
            result = super().export(spans)
        except Exception as e:
            self._report(False, f"{type(e).__name__}: {e}")
            return SpanExportResult.FAILURE
        self._report(
            result == SpanExportResult.SUCCESS,
            None
            if result == SpanExportResult.SUCCESS
            else "OTLP export returned FAILURE",
        )
        return result

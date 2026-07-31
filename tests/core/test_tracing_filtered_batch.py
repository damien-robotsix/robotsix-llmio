"""_FilteredBatchSpanProcessor unit tests — driven with _ListSpanExporter
and fake spans, no real OTel exporter."""

from __future__ import annotations

import pytest

from robotsix_llmio.core import tracing
from robotsix_llmio.core.tracing import LANGFUSE_PUBLIC_KEY
from tests.core._fake_span import _FakeSpan


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

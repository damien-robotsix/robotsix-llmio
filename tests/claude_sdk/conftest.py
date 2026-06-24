"""Shared fixtures for claude_sdk tests."""

from __future__ import annotations

import pytest


@pytest.fixture
def otel_exporter_tracer():
    """Set up an isolated OTel recording provider that routes spans to an
    InMemorySpanExporter so tests can inspect finished spans offline.
    Yields ``(exporter, tracer)``."""
    pytest.importorskip("opentelemetry.sdk")
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import SimpleSpanProcessor
    from opentelemetry.sdk.trace.export.in_memory_span_exporter import (
        InMemorySpanExporter,
    )

    exporter = InMemorySpanExporter()
    provider_obj = TracerProvider()
    provider_obj.add_span_processor(SimpleSpanProcessor(exporter))
    tracer = provider_obj.get_tracer("test")
    yield exporter, tracer

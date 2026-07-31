"""Fake OpenTelemetry span helpers shared across tracing processor tests."""

from __future__ import annotations


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

"""Offline tests for the shared :mod:`robotsix_llmio.logging` helper.

These run without OpenTelemetry and without a live span: the active span is
faked via ``monkeypatch`` on ``get_recording_span``. Each test operates on a
uniquely-named logger and tears down its handlers so global logging state does
not leak between tests.
"""

from __future__ import annotations

import io
import json
import logging

import pytest
import structlog

from robotsix_llmio import logging as llmio_logging
from robotsix_llmio.logging import (
    OTelTraceFilter,
    add_otel_trace_id,
    bind_correlation_id,
    setup_logging,
    setup_structlog,
)


@pytest.fixture
def logger_name(request):
    """A unique logger name, with its handlers cleaned up after the test."""
    name = f"test_logging_{request.node.name}"
    yield name
    target = logging.getLogger(name)
    for handler in list(target.handlers):
        target.removeHandler(handler)
    target.setLevel(logging.NOTSET)
    target.propagate = True


class _FakeSpanContext:
    def __init__(self, trace_id: int) -> None:
        self.trace_id = trace_id


class _FakeSpan:
    def __init__(self, trace_id: int) -> None:
        self._ctx = _FakeSpanContext(trace_id)

    def get_span_context(self):
        return self._ctx


def _configured_handler(name: str) -> logging.Handler:
    handlers = logging.getLogger(name).handlers
    assert len(handlers) == 1
    return handlers[0]


def test_import_has_no_top_level_opentelemetry():
    """The module source must not import opentelemetry at the top level."""
    import inspect

    source = inspect.getsource(llmio_logging)
    assert "import opentelemetry" not in source


def test_single_handler_with_filter_and_idempotent(logger_name):
    """One StreamHandler carrying an OTelTraceFilter; repeat calls add none."""
    setup_logging(loggers=(logger_name,))
    handler = _configured_handler(logger_name)
    assert isinstance(handler, logging.StreamHandler)
    assert any(isinstance(f, OTelTraceFilter) for f in handler.filters)

    setup_logging(loggers=(logger_name,))
    assert len(logging.getLogger(logger_name).handlers) == 1


def test_no_span_uses_placeholder(logger_name, monkeypatch):
    """Without an active span, emitting does not raise and uses ``-``."""
    monkeypatch.setattr(llmio_logging, "get_recording_span", lambda: None)
    record = logging.LogRecord(logger_name, logging.INFO, __file__, 1, "hi", None, None)
    assert OTelTraceFilter().filter(record) is True
    assert record.trace_id == "-"


def test_active_span_sets_hex_trace_id(logger_name, monkeypatch):
    """A fake span yields the 32-hex trace id on the record."""
    monkeypatch.setattr(llmio_logging, "get_recording_span", lambda: _FakeSpan(0x1234))
    record = logging.LogRecord(logger_name, logging.INFO, __file__, 1, "hi", None, None)
    OTelTraceFilter().filter(record)
    assert record.trace_id == format(0x1234, "032x")


def test_json_format_is_valid(logger_name, monkeypatch):
    """``fmt='json'`` produces one parseable JSON object with required keys."""
    monkeypatch.setattr(llmio_logging, "get_recording_span", lambda: None)
    stream = io.StringIO()
    setup_logging(loggers=(logger_name,), fmt="json", stream=stream)
    logging.getLogger(logger_name).info("hello json")
    payload = json.loads(stream.getvalue().strip())
    assert payload["level"] == "INFO"
    assert payload["logger"] == logger_name
    assert payload["message"] == "hello json"
    assert payload["trace_id"] == "-"


def test_json_format_via_env(logger_name, monkeypatch):
    """``LOG_FORMAT=json`` selects the JSON formatter."""
    monkeypatch.setattr(llmio_logging, "get_recording_span", lambda: None)
    monkeypatch.setenv("LOG_FORMAT", "json")
    stream = io.StringIO()
    setup_logging(loggers=(logger_name,), stream=stream)
    logging.getLogger(logger_name).info("env json")
    assert json.loads(stream.getvalue().strip())["message"] == "env json"


def test_json_format_includes_exception(logger_name, monkeypatch):
    """An exc_info record renders the traceback in the JSON payload."""
    monkeypatch.setattr(llmio_logging, "get_recording_span", lambda: None)
    stream = io.StringIO()
    setup_logging(loggers=(logger_name,), fmt="json", stream=stream)
    try:
        raise ValueError("boom")
    except ValueError:
        logging.getLogger(logger_name).exception("failed")
    payload = json.loads(stream.getvalue().strip())
    assert "boom" in payload["exc_info"]


def test_console_format_includes_trace_token(logger_name, monkeypatch):
    """``fmt='console'`` renders the ``[<trace_id>]`` token."""
    monkeypatch.setattr(llmio_logging, "get_recording_span", lambda: None)
    stream = io.StringIO()
    setup_logging(loggers=(logger_name,), fmt="console", stream=stream)
    logging.getLogger(logger_name).info("console line")
    assert "[-]" in stream.getvalue()
    assert "console line" in stream.getvalue()


def test_text_alias_via_env(logger_name, monkeypatch):
    """``LOG_FORMAT=console`` selects the text formatter."""
    monkeypatch.setattr(llmio_logging, "get_recording_span", lambda: None)
    monkeypatch.setenv("LOG_FORMAT", "console")
    stream = io.StringIO()
    setup_logging(loggers=(logger_name,), stream=stream)
    logging.getLogger(logger_name).info("text line")
    assert "[-]" in stream.getvalue()


def test_unrecognized_format_falls_back_to_console(logger_name, monkeypatch):
    """An unknown format does not raise and falls back to console text."""
    monkeypatch.setattr(llmio_logging, "get_recording_span", lambda: None)
    stream = io.StringIO()
    setup_logging(loggers=(logger_name,), fmt="bogus", stream=stream)
    logging.getLogger(logger_name).info("fallback")
    assert "[-]" in stream.getvalue()


def test_level_explicit_arg(logger_name):
    """``level='DEBUG'`` sets the configured logger to DEBUG."""
    setup_logging(loggers=(logger_name,), level="DEBUG")
    assert logging.getLogger(logger_name).level == logging.DEBUG


def test_level_int_arg(logger_name):
    """An int level is applied directly."""
    setup_logging(loggers=(logger_name,), level=logging.WARNING)
    assert logging.getLogger(logger_name).level == logging.WARNING


def test_level_from_env(logger_name, monkeypatch):
    """``LOG_LEVEL=DEBUG`` sets the configured logger to DEBUG."""
    monkeypatch.setenv("LOG_LEVEL", "DEBUG")
    setup_logging(loggers=(logger_name,))
    assert logging.getLogger(logger_name).level == logging.DEBUG


def test_level_explicit_beats_env(logger_name, monkeypatch):
    """Explicit ``level`` takes precedence over ``LOG_LEVEL``."""
    monkeypatch.setenv("LOG_LEVEL", "DEBUG")
    setup_logging(loggers=(logger_name,), level="ERROR")
    assert logging.getLogger(logger_name).level == logging.ERROR


def test_level_defaults_to_info(logger_name, monkeypatch):
    """With no arg and no env, the level defaults to INFO."""
    monkeypatch.delenv("LOG_LEVEL", raising=False)
    setup_logging(loggers=(logger_name,))
    assert logging.getLogger(logger_name).level == logging.INFO


def test_multiple_loggers_configured(monkeypatch):
    """Multiple logger names are each configured (no hardcoded names)."""
    names = ("test_logging_app", "test_logging_lib")
    try:
        setup_logging(loggers=names, level="DEBUG")
        for name in names:
            target = logging.getLogger(name)
            assert target.level == logging.DEBUG
            assert len(target.handlers) == 1
    finally:
        for name in names:
            target = logging.getLogger(name)
            for handler in list(target.handlers):
                target.removeHandler(handler)
            target.setLevel(logging.NOTSET)


def test_trace_filter_tolerates_a_partial_span(monkeypatch):
    """A recording span without ``get_span_context`` must not break logging.

    Anything reporting itself as recording reaches this filter — span shims,
    no-op spans, and test doubles that implement only the slice of the OTel
    Span protocol their own caller needs. A filter that raises takes down
    logging for the entire process, so a missing attribute has to degrade to
    "no trace id", not to an exception.
    """
    import robotsix_llmio.logging as llmio_logging

    class PartialSpan:
        """Implements is_recording/set_attribute only — no get_span_context."""

        def is_recording(self):
            return True

        def set_attribute(self, key, value):
            pass

    monkeypatch.setattr(llmio_logging, "get_recording_span", lambda: PartialSpan())

    record = logging.LogRecord(
        name="t",
        level=logging.INFO,
        pathname=__file__,
        lineno=1,
        msg="hello",
        args=(),
        exc_info=None,
    )
    assert OTelTraceFilter().filter(record) is True
    assert record.trace_id == "-"


@pytest.fixture
def structlog_reset():
    """Restore global root-logger and structlog config after a structlog test."""
    root = logging.getLogger()
    saved_handlers = list(root.handlers)
    saved_level = root.level
    yield
    structlog.contextvars.clear_contextvars()
    structlog.reset_defaults()
    for handler in list(root.handlers):
        if handler not in saved_handlers:
            root.removeHandler(handler)
    root.setLevel(saved_level)


def test_structlog_json_renders_stdlib_record(
    logger_name, monkeypatch, structlog_reset
):
    """A foreign stdlib record renders as JSON through the root bridge."""
    monkeypatch.setattr(llmio_logging, "get_recording_span", lambda: None)
    stream = io.StringIO()
    setup_structlog(loggers=(logger_name,), fmt="json", stream=stream)
    logging.getLogger(logger_name).info("hello structlog")
    payload = json.loads(stream.getvalue().strip())
    assert payload["event"] == "hello structlog"
    assert payload["level"] == "info"
    assert payload["logger"] == logger_name
    assert payload["trace_id"] == "-"
    assert "timestamp" in payload


def test_structlog_json_renders_native_logger(
    logger_name, monkeypatch, structlog_reset
):
    """A structlog-native logger renders through the same root bridge."""
    monkeypatch.setattr(llmio_logging, "get_recording_span", lambda: None)
    stream = io.StringIO()
    setup_structlog(loggers=(logger_name,), fmt="json", stream=stream)
    structlog.get_logger(logger_name).info("native event", extra_key="v")
    payload = json.loads(stream.getvalue().strip())
    assert payload["event"] == "native event"
    assert payload["extra_key"] == "v"
    assert payload["trace_id"] == "-"


def test_structlog_console_fmt_is_human_readable(
    logger_name, monkeypatch, structlog_reset
):
    """``fmt='console'`` produces a non-JSON line containing the message."""
    monkeypatch.setattr(llmio_logging, "get_recording_span", lambda: None)
    stream = io.StringIO()
    setup_structlog(loggers=(logger_name,), fmt="console", stream=stream)
    logging.getLogger(logger_name).info("console structlog")
    output = stream.getvalue()
    assert "console structlog" in output
    with pytest.raises(json.JSONDecodeError):
        json.loads(output.strip())


def test_structlog_trace_id_from_active_span(logger_name, monkeypatch, structlog_reset):
    """An active span stamps the 32-hex trace id onto the JSON event."""
    monkeypatch.setattr(llmio_logging, "get_recording_span", lambda: _FakeSpan(0xABCD))
    stream = io.StringIO()
    setup_structlog(loggers=(logger_name,), fmt="json", stream=stream)
    logging.getLogger(logger_name).info("traced")
    payload = json.loads(stream.getvalue().strip())
    assert payload["trace_id"] == format(0xABCD, "032x")


def test_structlog_correlation_id_disabled_by_default(
    logger_name, monkeypatch, structlog_reset
):
    """Without ``correlation_id=True`` a bound id is not merged onto events."""
    monkeypatch.setattr(llmio_logging, "get_recording_span", lambda: None)
    stream = io.StringIO()
    setup_structlog(loggers=(logger_name,), fmt="json", stream=stream)
    bind_correlation_id("req-123")
    logging.getLogger(logger_name).info("no cid")
    payload = json.loads(stream.getvalue().strip())
    assert "correlation_id" not in payload


def test_structlog_correlation_id_merged_when_enabled(
    logger_name, monkeypatch, structlog_reset
):
    """``correlation_id=True`` merges the bound id, and ``None`` clears it."""
    monkeypatch.setattr(llmio_logging, "get_recording_span", lambda: None)
    stream = io.StringIO()
    setup_structlog(
        loggers=(logger_name,), fmt="json", stream=stream, correlation_id=True
    )
    bind_correlation_id("req-456")
    logging.getLogger(logger_name).info("with cid")
    first = json.loads(stream.getvalue().strip().splitlines()[0])
    assert first["correlation_id"] == "req-456"

    bind_correlation_id(None)
    stream.truncate(0)
    stream.seek(0)
    logging.getLogger(logger_name).info("cleared cid")
    assert "correlation_id" not in json.loads(stream.getvalue().strip())


def test_structlog_level_from_env(logger_name, monkeypatch, structlog_reset):
    """``LOG_LEVEL`` drives both the root and the named logger level."""
    monkeypatch.setenv("LOG_LEVEL", "WARNING")
    setup_structlog(loggers=(logger_name,))
    assert logging.getLogger().level == logging.WARNING
    assert logging.getLogger(logger_name).level == logging.WARNING


def test_structlog_is_idempotent(logger_name, monkeypatch, structlog_reset):
    """Repeat calls reuse the single marked root handler."""
    monkeypatch.setattr(llmio_logging, "get_recording_span", lambda: None)
    setup_structlog(loggers=(logger_name,), fmt="json")
    root = logging.getLogger()
    marked = [
        h for h in root.handlers if getattr(h, "_robotsix_llmio_configured", False)
    ]
    assert len(marked) == 1
    setup_structlog(loggers=(logger_name,), fmt="json")
    marked_again = [
        h for h in root.handlers if getattr(h, "_robotsix_llmio_configured", False)
    ]
    assert len(marked_again) == 1


def test_structlog_replaces_stdlib_handler_on_named_logger(
    logger_name, monkeypatch, structlog_reset
):
    """A prior ``setup_logging`` handler is dropped so records don't double-emit."""
    monkeypatch.setattr(llmio_logging, "get_recording_span", lambda: None)
    setup_logging(loggers=(logger_name,))
    assert len(logging.getLogger(logger_name).handlers) == 1
    stream = io.StringIO()
    setup_structlog(loggers=(logger_name,), fmt="json", stream=stream)
    target = logging.getLogger(logger_name)
    assert target.handlers == []
    assert target.propagate is True
    target.info("single line")
    assert len(stream.getvalue().strip().splitlines()) == 1


def test_structlog_json_interpolates_positional_args(
    logger_name, monkeypatch, structlog_reset
):
    """%-style call sites interpolate; no raw ``positional_args`` key leaks.

    Regression for the missing ``PositionalArgumentsFormatter`` in the shared
    chain: without it, ``logger.info("warming (%d projects, %d windows)", 3, 5)``
    rendered the literal ``%d`` tokens plus a raw ``positional_args`` list.
    """
    monkeypatch.setattr(llmio_logging, "get_recording_span", lambda: None)
    stream = io.StringIO()
    setup_structlog(loggers=(logger_name,), fmt="json", stream=stream)
    logging.getLogger(logger_name).info("warming (%d projects, %d windows)", 3, 5)
    payload = json.loads(stream.getvalue().strip())
    assert payload["event"] == "warming (3 projects, 5 windows)"
    assert "positional_args" not in payload
    assert "%d" not in payload["event"]


def test_structlog_console_interpolates_positional_args(
    logger_name, monkeypatch, structlog_reset
):
    """%-style call sites interpolate in console output too (JSON/console parity)."""
    monkeypatch.setattr(llmio_logging, "get_recording_span", lambda: None)
    stream = io.StringIO()
    setup_structlog(loggers=(logger_name,), fmt="console", stream=stream)
    logging.getLogger(logger_name).info("warming (%d projects, %d windows)", 3, 5)
    output = stream.getvalue()
    assert "warming (3 projects, 5 windows)" in output
    assert "%d" not in output
    assert "positional_args=" not in output


def test_add_otel_trace_id_processor_tolerates_partial_span(monkeypatch):
    """The structlog trace-id processor degrades to ``-`` on a partial span."""

    class PartialSpan:
        def is_recording(self):
            return True

    monkeypatch.setattr(llmio_logging, "get_recording_span", lambda: PartialSpan())
    event = add_otel_trace_id(None, "info", {"event": "x"})
    assert event["trace_id"] == "-"

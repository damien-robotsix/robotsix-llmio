"""Shared, stdlib-only logging setup with OTel trace-id injection.

``robotsix_llmio`` owns the OpenTelemetry→Langfuse export layer, so the
bridge that stamps the active trace id onto log records belongs here too.
This module provides a small, idempotent :func:`setup_logging` helper that
both ``robotsix_llmio`` and its downstream consumers can share instead of
reinventing the same boilerplate.

The module imports cleanly even when the ``tracing`` extra (OpenTelemetry)
is not installed: the trace id is obtained via
:func:`robotsix_llmio.core.get_recording_span`, which guards its own
``ImportError`` and returns ``None`` when OpenTelemetry is absent.

``filters: ["!^_"]`` keeps underscore-prefixed names private in the docs.
"""

from __future__ import annotations

import json
import logging
import os
import sys
from collections.abc import Sequence
from typing import TextIO

from robotsix_llmio.core import get_recording_span

#: Placeholder trace id used when no recording span is active.
_NO_TRACE_ID = "-"

#: Marker attribute set on handlers created by :func:`setup_logging` so a
#: repeat call can detect and reuse the existing handler (idempotency).
_CONFIGURED_MARKER = "_robotsix_llmio_configured"

#: Text format used for the ``"console"`` / ``"text"`` format.
_CONSOLE_FORMAT = "%(asctime)s %(levelname)s %(name)s [%(trace_id)s] %(message)s"


class OTelTraceFilter(logging.Filter):
    """Stamp the active OpenTelemetry trace id onto every log record.

    The filter sets ``record.trace_id`` to the 32-hex-char id of the active
    recording span, or to ``"-"`` when no span is active (or OpenTelemetry is
    not installed). It always returns ``True`` and never raises.
    """

    def filter(self, record: logging.LogRecord) -> bool:
        """Set ``record.trace_id`` and keep the record.

        Args:
            record: The log record to annotate.

        Returns:
            Always ``True`` so the record is never dropped.
        """
        span = get_recording_span()
        if span is not None:
            tid = getattr(span.get_span_context(), "trace_id", 0)
            record.trace_id = format(tid, "032x") if tid else _NO_TRACE_ID
        else:
            record.trace_id = _NO_TRACE_ID
        return True


class _JsonFormatter(logging.Formatter):
    """Render each log record as a single line of JSON via stdlib ``json``."""

    def format(self, record: logging.LogRecord) -> str:
        """Format *record* as a one-line JSON object.

        Args:
            record: The log record to render.

        Returns:
            A JSON string parseable by :func:`json.loads`.
        """
        payload: dict[str, object] = {
            "timestamp": self.formatTime(record),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "trace_id": getattr(record, "trace_id", _NO_TRACE_ID),
        }
        if record.exc_info:
            payload["exc_info"] = self.formatException(record.exc_info)
        return json.dumps(payload)


def _resolve_level(level: int | str | None) -> int:
    """Resolve the effective log level.

    First non-``None`` wins: explicit *level* → env ``LOG_LEVEL`` → ``"INFO"``.
    Accepts a level name (case-insensitive) or an int.

    Args:
        level: Explicit level, or ``None`` to fall back to env/default.

    Returns:
        The resolved numeric logging level.
    """
    resolved: int | str = (
        level if level is not None else os.environ.get("LOG_LEVEL", "INFO")
    )
    if isinstance(resolved, int):
        return resolved
    return logging.getLevelNamesMapping().get(resolved.upper(), logging.INFO)


def _resolve_formatter(fmt: str | None) -> logging.Formatter:
    """Resolve the formatter from *fmt*.

    First non-``None`` wins: explicit *fmt* → env ``LOG_FORMAT`` → ``"console"``.
    ``"json"`` selects the JSON formatter; ``"console"``/``"text"`` (and any
    unrecognized value) selects the text formatter.

    Args:
        fmt: Explicit format name, or ``None`` to fall back to env/default.

    Returns:
        The resolved :class:`logging.Formatter`.
    """
    name = fmt if fmt is not None else os.environ.get("LOG_FORMAT", "console")
    if name.lower() == "json":
        return _JsonFormatter()
    return logging.Formatter(_CONSOLE_FORMAT)


def setup_logging(
    *,
    level: int | str | None = None,
    fmt: str | None = None,
    loggers: Sequence[str] = ("robotsix_llmio",),
    stream: TextIO | None = None,
) -> None:
    """Configure application logging with OTel trace-id injection (idempotent).

    Attaches a single :class:`logging.StreamHandler` — carrying an
    :class:`OTelTraceFilter` and the resolved formatter — to each named
    logger, and sets each logger's level. The root logger is never touched.

    Args:
        level: Explicit level (name or int). Falls back to ``LOG_LEVEL`` env,
            then ``"INFO"``.
        fmt: ``"console"``/``"text"`` or ``"json"``. Falls back to
            ``LOG_FORMAT`` env, then ``"console"``. Unrecognized values fall
            back to ``"console"``.
        loggers: Logger names to configure. Defaults to llmio's own
            namespace; consumers pass their app namespaces too.
        stream: Target stream. Defaults to :data:`sys.stdout`.

    Returns:
        ``None``.
    """
    resolved_level = _resolve_level(level)
    formatter = _resolve_formatter(fmt)
    target_stream = stream if stream is not None else sys.stdout

    for name in loggers:
        target = logging.getLogger(name)
        target.setLevel(resolved_level)
        target.propagate = False

        existing = next(
            (h for h in target.handlers if getattr(h, _CONFIGURED_MARKER, False)),
            None,
        )
        if existing is not None:
            existing.setFormatter(formatter)
            existing.setLevel(resolved_level)
            continue

        handler: logging.Handler = logging.StreamHandler(target_stream)
        handler.addFilter(OTelTraceFilter())
        handler.setFormatter(formatter)
        handler.setLevel(resolved_level)
        setattr(handler, _CONFIGURED_MARKER, True)
        target.addHandler(handler)

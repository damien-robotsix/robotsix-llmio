"""Capture the Claude CLI's stderr so failures explain themselves.

The SDK spawns the CLI as a subprocess. When it exits non-zero the resulting
``ProcessError`` carries only ``Command failed with exit code 1`` and the
placeholder ``stderr='Check stderr output for details'`` — the real reason is
written to the CLI's own stderr and never reaches the exception. Every caller
that wrapped that error therefore reported something unactionable.

Two failures on robotsix-chat were diagnosable only by reading the container
log by hand: ``Session ID <uuid> is already in use`` and ``No conversation
found with session ID: <uuid>``. Both name their own fix; neither was visible
to the code deciding whether to retry.

``ClaudeAgentOptions`` accepts a ``stderr`` callback. This module wires one that
keeps the last few lines in a context-local buffer, so the wrapper can append
them to the error it raises. Context-local rather than global because agent runs
are concurrent — a buffer shared across tasks would attribute one run's stderr
to another's failure.
"""

from __future__ import annotations

import logging
from contextvars import ContextVar

logger = logging.getLogger(__name__)

#: Kept small on purpose: the CLI's diagnosis is the last line or two, and the
#: buffer is echoed into an exception message that ends up in logs.
_MAX_LINES = 20

_buffer: ContextVar[list[str] | None] = ContextVar("claude_cli_stderr", default=None)


def start_capture() -> None:
    """Begin capturing CLI stderr for the current context."""
    _buffer.set([])


def record(line: str) -> None:
    """Record one CLI stderr *line*, keeping only the most recent ones."""
    text = line.rstrip()
    if not text:
        return
    logger.debug("claude cli stderr: %s", text)
    buf = _buffer.get()
    if buf is None:
        return
    buf.append(text)
    if len(buf) > _MAX_LINES:
        del buf[: len(buf) - _MAX_LINES]


def captured() -> str:
    """Return the captured stderr for the current context, newest last."""
    buf = _buffer.get()
    return "\n".join(buf) if buf else ""


def describe() -> str:
    """Return a suffix naming the CLI's stderr, or ``""`` when nothing useful.

    The SDK's own placeholder is filtered out — repeating "check stderr output
    for details" inside an error message is what made these failures opaque in
    the first place.
    """
    text = captured()
    if not text or text.lower().startswith("check stderr output"):
        return ""
    return f"\nCLI stderr:\n{text}"

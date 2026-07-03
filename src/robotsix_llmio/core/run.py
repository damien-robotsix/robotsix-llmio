"""Canonical agent-run lifecycle: trace span + bounded retry + deterministic
``close()``.

Composes the existing primitives — :func:`start_trace`, the bounded-retry
helpers in :mod:`.retry`, and an :class:`AgentHandle` — into the one shape every
consumer needs: open a trace span, run the (caller-supplied) model call under
bounded transient/rate-limit retry, and always ``close()`` the handle in a
``finally`` so the HTTP client is released even when the run raises.

Provider-agnostic: the helper operates on a *pre-built* handle plus a caller
callable; it never builds the agent itself. The only thing required of *handle*
is a synchronous ``close()``, so both :class:`AgentHandle` and the claude-sdk
tool handle work.
"""

from __future__ import annotations

import asyncio
import time
from collections.abc import Awaitable, Callable
from typing import Any, TypeVar

from .agent import AgentHandle
from .retry import (
    acall_with_retry,
    acall_with_retry_and_fallback,
    call_with_retry,
    call_with_retry_and_fallback,
    is_transient,
)
from .tracing import start_trace

T = TypeVar("T")


def run_agent[T](
    handle: AgentHandle,
    run: Callable[[], T],
    *,
    label: str,
    session_id: str | None = None,
    project: str | None = None,
    fallback: Callable[[], T] | None = None,
    what: str = "model call",
    trace_input: Any = None,
    sleep: Callable[[float], None] = time.sleep,
    is_transient_fn: Callable[[BaseException], bool] = is_transient,
) -> T:
    """Run *run* under a trace span and bounded retry, always closing *handle*.

    Opens a root trace span named *label* (optionally grouped under *session_id*
    and routed to *project*). When *trace_input* is not ``None`` it is recorded
    as the span input. *run* is executed under :func:`call_with_retry` (or
    :func:`call_with_retry_and_fallback` when *fallback* is given); on success
    the result is recorded as the span output and returned. ``handle.close()``
    runs in a ``finally`` — even when *run* raises after retries are exhausted.
    """
    with start_trace(label, session_id=session_id, project=project) as span:
        try:
            if trace_input is not None:
                span.set_input(trace_input)
            if fallback is not None:
                result = call_with_retry_and_fallback(
                    run,
                    fallback,
                    what=what,
                    sleep=sleep,
                    is_transient_primary=is_transient_fn,
                    is_transient_fallback=is_transient_fn,
                )
            else:
                result = call_with_retry(
                    run, what=what, sleep=sleep, is_transient_fn=is_transient_fn
                )
            span.set_output(result)
            return result
        finally:
            handle.close()


async def arun_agent[T](
    handle: AgentHandle,
    run: Callable[[], Awaitable[T]],
    *,
    label: str,
    session_id: str | None = None,
    project: str | None = None,
    fallback: Callable[[], Awaitable[T]] | None = None,
    what: str = "model call",
    trace_input: Any = None,
    sleep: Callable[[float], Awaitable[None]] = asyncio.sleep,
    is_transient_fn: Callable[[BaseException], bool] = is_transient,
) -> T:
    """Async mirror of :func:`run_agent`.

    Same composition, except *run*/*fallback* are awaited and the retry uses
    :func:`acall_with_retry` / :func:`acall_with_retry_and_fallback` with
    *sleep* defaulting to :func:`asyncio.sleep`. ``handle.aclose()`` is
    awaited in the ``finally`` block to close the HTTP client in the caller's
    running event loop.
    """
    with start_trace(label, session_id=session_id, project=project) as span:
        try:
            if trace_input is not None:
                span.set_input(trace_input)
            if fallback is not None:
                result = await acall_with_retry_and_fallback(
                    run,
                    fallback,
                    what=what,
                    sleep=sleep,
                    is_transient_primary=is_transient_fn,
                    is_transient_fallback=is_transient_fn,
                )
            else:
                result = await acall_with_retry(
                    run, what=what, sleep=sleep, is_transient_fn=is_transient_fn
                )
            span.set_output(result)
            return result
        finally:
            await handle.aclose()

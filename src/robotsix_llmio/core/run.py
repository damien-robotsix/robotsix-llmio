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


async def _run_with_trace_and_close(
    run_fn: Callable[[], Any],
    close_fn: Callable[[], Any],
    *,
    invoke_run: Callable[[Callable[[], Any]], Awaitable[Any]],
    invoke_close: Callable[[Callable[[], Any]], Awaitable[None]],
    label: str,
    session_id: str | None = None,
    project: str | None = None,
    trace_input: Any = None,
) -> Any:
    """Open a trace span, run *run_fn*, always close via *invoke_close*.

    *invoke_run* and *invoke_close* adapt sync vs async — they are the
    only injection points needed to share the trace+close control flow
    between :func:`run_agent` and :func:`arun_agent`.
    """
    with start_trace(label, session_id=session_id, project=project) as span:
        try:
            if trace_input is not None:
                span.set_input(trace_input)
            result = await invoke_run(run_fn)
            span.set_output(result)
            return result
        finally:
            await invoke_close(close_fn)


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

    def _run() -> T:
        if fallback is not None:
            return call_with_retry_and_fallback(
                run,
                fallback,
                what=what,
                sleep=sleep,
                is_transient_primary=is_transient_fn,
                is_transient_fallback=is_transient_fn,
            )
        return call_with_retry(
            run, what=what, sleep=sleep, is_transient_fn=is_transient_fn
        )

    async def _invoke_run(f: Callable[[], Any]) -> Any:
        return f()

    async def _invoke_close(f: Callable[[], Any]) -> None:
        f()

    return asyncio.run(
        _run_with_trace_and_close(
            _run,
            handle.close,
            invoke_run=_invoke_run,
            invoke_close=_invoke_close,
            label=label,
            session_id=session_id,
            project=project,
            trace_input=trace_input,
        )
    )


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

    async def _run() -> T:
        if fallback is not None:
            return await acall_with_retry_and_fallback(
                run,
                fallback,
                what=what,
                sleep=sleep,
                is_transient_primary=is_transient_fn,
                is_transient_fallback=is_transient_fn,
            )
        return await acall_with_retry(
            run, what=what, sleep=sleep, is_transient_fn=is_transient_fn
        )

    async def _invoke_run(f: Callable[[], Any]) -> Any:
        return await f()

    async def _invoke_close(f: Callable[[], Any]) -> None:
        await f()

    return await _run_with_trace_and_close(
        _run,
        handle.aclose,
        invoke_run=_invoke_run,
        invoke_close=_invoke_close,
        label=label,
        session_id=session_id,
        project=project,
        trace_input=trace_input,
    )

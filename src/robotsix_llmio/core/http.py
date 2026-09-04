"""Timeout-bounded async HTTP client for provider SDKs."""

from __future__ import annotations

import asyncio
import weakref
from typing import TYPE_CHECKING, Any

from . import constants

if TYPE_CHECKING:
    # httpx2 is pydantic-ai 2.x's HTTP layer; a legacy ``httpx.AsyncClient``
    # handed to a 2.x provider is deprecated (PydanticAIDeprecationWarning,
    # removal in v3), so the provider-facing client uses the fork.
    import httpx2


# Strong refs to in-flight aclose() tasks scheduled on a running loop, so
# they aren't garbage-collected mid-close (asyncio holds only weak refs).
_pending_closes: set[Any] = set()


def _drop_coroutine(coro: Any) -> None:
    """Consume a coroutine that can no longer be awaited, so it doesn't emit
    a "never awaited" RuntimeWarning when garbage-collected."""
    close = getattr(coro, "close", None)
    if close is not None:
        close()


def _close_async_client(client: Any) -> None:
    """Close an httpx2.AsyncClient from a GC/finalize context.

    The finalizer usually fires in a thread whose event loop is RUNNING (GC
    triggers inside the service's own loop): there ``run_until_complete`` on
    a fresh loop raises RuntimeError, so the close is scheduled as a task on
    the running loop instead — the old swallow-and-drop path left the
    ``aclose()`` coroutine un-awaited (a RuntimeWarning per collected client
    and an unclosed transport). Without a running loop, a temporary loop
    drives the close synchronously. Errors never raise out of finalize.
    """
    coro = client.aclose()
    try:
        running = asyncio.get_running_loop()
    except RuntimeError:
        running = None
    if running is not None:
        try:
            task = running.create_task(coro)
        except RuntimeError:
            # Loop is shutting down — nothing can await this anymore.
            _drop_coroutine(coro)
        else:
            _pending_closes.add(task)
            task.add_done_callback(_pending_closes.discard)
        return
    try:
        loop = asyncio.new_event_loop()
        try:
            loop.run_until_complete(coro)
        finally:
            loop.close()
    except RuntimeError, OSError:
        # Expected event-loop/transport teardown errors during GC/finalize
        # (loop-state RuntimeError, socket-close OSError) are safe to ignore;
        # other exception types (e.g. AttributeError/TypeError from a broken
        # aclose() call) propagate so genuine bugs aren't masked.
        _drop_coroutine(coro)


def timeout_http_client() -> httpx2.AsyncClient:
    """A fresh ``httpx2.AsyncClient`` with a hard per-request timeout, so a
    hung/glacial provider connection raises instead of blocking forever.
    Pass to the provider as its ``http_client``.

    httpx2 is pydantic-ai 2.x's HTTP layer: a legacy ``httpx.AsyncClient``
    handed to a 2.x provider emits ``PydanticAIDeprecationWarning`` (and is
    slated for removal in v3), so the provider-facing client is built on the
    fork.
    """
    import httpx2

    client = httpx2.AsyncClient(
        timeout=httpx2.Timeout(
            constants.MODEL_REQUEST_TIMEOUT, connect=constants.CONNECT_TIMEOUT
        )
    )
    _ref = weakref.ref(client)

    def _gc_close() -> None:
        c = _ref()
        if c is not None:
            _close_async_client(c)

    weakref.finalize(client, _gc_close)
    return client

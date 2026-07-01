"""Timeout-bounded async HTTP client for provider SDKs."""

from __future__ import annotations

import asyncio
import weakref
from typing import TYPE_CHECKING, Any

from . import constants

if TYPE_CHECKING:
    import httpx


def _close_async_client(client: Any) -> None:
    """Close an httpx.AsyncClient from outside its original event loop.

    Creates a temporary event loop to run aclose(), swallowing errors so
    cleanup never raises in a finally/__del__ context.
    """
    try:
        loop = asyncio.new_event_loop()
        try:
            loop.run_until_complete(client.aclose())
        finally:
            loop.close()
    except (RuntimeError, OSError):
        # Expected event-loop/transport teardown errors during GC/finalize
        # (loop-state RuntimeError, socket-close OSError) are safe to ignore;
        # other exception types (e.g. AttributeError/TypeError from a broken
        # aclose() call) propagate so genuine bugs aren't masked.
        pass


def timeout_http_client() -> httpx.AsyncClient:
    """A fresh ``httpx.AsyncClient`` with a hard per-request timeout, so a
    hung/glacial provider connection raises instead of blocking forever.
    Pass to the provider as its ``http_client``.
    """
    import httpx

    client = httpx.AsyncClient(
        timeout=httpx.Timeout(
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

"""Sync wrapper loop-free enforcement tests."""

from __future__ import annotations

import asyncio

import pytest
from conftest import _HTTPErr

from robotsix_llmio.core.retry import (
    call_with_retry,
    call_with_retry_and_fallback,
)

# --- sync wrappers must run fn loop-free ---------------------------------
# The sync wrappers drive the shared async retry loop WITHOUT an event loop
# (via a sync coroutine driver).  This is load-bearing: the wrapped fn is
# often run_sync-style — pydantic-ai ``Agent.run_sync`` uses
# ``get_event_loop().run_until_complete()``, the claude_sdk tool agent uses
# ``asyncio.run()`` — and both raise RuntimeError when a loop is already
# running in the calling thread.


def test_call_with_retry_fn_sees_no_running_loop():
    def fn():
        with pytest.raises(RuntimeError):
            asyncio.get_running_loop()
        return "ok"

    assert call_with_retry(fn, sleep=lambda _: None) == "ok"


def test_call_with_retry_supports_run_sync_style_fn():
    async def payload():
        return "ok"

    def fn():
        # claude_sdk tool-agent style: bare asyncio.run inside the callable.
        return asyncio.run(payload())

    assert call_with_retry(fn, sleep=lambda _: None) == "ok"


def test_call_with_retry_supports_run_until_complete_style_fn():
    async def payload():
        return "ok"

    def fn():
        # pydantic-ai run_sync style: fresh loop + run_until_complete.
        loop = asyncio.new_event_loop()
        try:
            return loop.run_until_complete(payload())
        finally:
            loop.close()

    assert call_with_retry(fn, sleep=lambda _: None) == "ok"


def test_call_with_retry_run_sync_style_fn_retries_transient():
    calls = {"n": 0}

    async def payload():
        calls["n"] += 1
        if calls["n"] < 3:
            raise _HTTPErr(503)
        return "ok"

    def fn():
        return asyncio.run(payload())

    assert call_with_retry(fn, sleep=lambda _: None) == "ok"
    assert calls["n"] == 3


def test_call_with_retry_and_fallback_supports_run_sync_style_fn():
    async def boom():
        raise _HTTPErr(503)

    async def payload():
        return "fallback-ok"

    def primary():
        return asyncio.run(boom())

    def fallback():
        return asyncio.run(payload())

    assert (
        call_with_retry_and_fallback(primary, fallback, sleep=lambda _: None)
        == "fallback-ok"
    )

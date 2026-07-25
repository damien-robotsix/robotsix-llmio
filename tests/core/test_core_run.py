"""Canonical agent-run lifecycle: trace span + bounded retry + deterministic
``handle.close()`` in a ``finally`` — for both the sync and async helpers.

Offline: ``start_trace`` yields a non-recording span (no Langfuse), so
``set_input``/``set_output`` are no-ops that must not raise. Sleeps are injected
no-ops so backoff never actually waits. There is no ``pytest-asyncio``; async
helpers are driven via ``asyncio.run(...)`` to match the repo convention.
"""

from __future__ import annotations

import asyncio

import pytest

from robotsix_llmio.core.run import arun_agent, run_agent


class _HTTPErr(Exception):
    def __init__(self, status_code):
        self.status_code = status_code


class _FakeHandle:
    """A handle that records how many times ``close()`` and ``aclose()``
    were called."""

    def __init__(self) -> None:
        self.closed = 0
        self.aclose_count = 0

    def close(self) -> None:
        self.closed += 1

    async def aclose(self) -> None:
        self.aclose_count += 1


def _noop_sleep(_d: float) -> None:
    return None


async def _anoop_sleep(_d: float) -> None:
    return None


# --- sync run_agent --------------------------------------------------------


def test_run_agent_happy_path_returns_and_closes_once():
    handle = _FakeHandle()
    out = run_agent(handle, lambda: "ok", label="t", sleep=_noop_sleep)
    assert out == "ok"
    assert handle.closed == 1


def test_run_agent_closes_when_run_raises_non_transient():
    handle = _FakeHandle()

    def run():
        raise _HTTPErr(400)  # non-transient → propagates immediately

    with pytest.raises(_HTTPErr):
        run_agent(handle, run, label="t", sleep=_noop_sleep)
    assert handle.closed == 1


def test_run_agent_retries_transient_then_succeeds():
    handle = _FakeHandle()
    calls = {"n": 0}

    def run():
        calls["n"] += 1
        if calls["n"] < 3:
            raise _HTTPErr(503)
        return "ok"

    out = run_agent(handle, run, label="t", sleep=_noop_sleep)
    assert out == "ok"
    assert calls["n"] == 3
    assert handle.closed == 1


def test_run_agent_uses_fallback_after_primary_exhausted():
    handle = _FakeHandle()
    calls = {"primary": 0, "fallback": 0}

    def primary():
        calls["primary"] += 1
        raise _HTTPErr(503)  # always transient → exhausts retries

    def fallback():
        calls["fallback"] += 1
        return "fallback-ok"

    out = run_agent(handle, primary, label="t", fallback=fallback, sleep=_noop_sleep)
    assert out == "fallback-ok"
    assert calls["primary"] > 1
    assert calls["fallback"] == 1
    assert handle.closed == 1


def test_run_agent_offline_trace_input_output_noop():
    """Offline start_trace → non-recording span; recording trace_input/output
    must not raise (mirror test_tracing_setup::test_start_trace_safe_without_provider)."""
    handle = _FakeHandle()
    out = run_agent(
        handle,
        lambda: "done",
        label="offline-trace",
        session_id="s",
        project="pk-x",
        trace_input={"a": 1},
        sleep=_noop_sleep,
    )
    assert out == "done"
    assert handle.closed == 1


# --- async arun_agent ------------------------------------------------------


def test_arun_agent_happy_path_returns_and_closes_once():
    handle = _FakeHandle()

    async def run():
        return "ok"

    out = asyncio.run(arun_agent(handle, run, label="t", sleep=_anoop_sleep))
    assert out == "ok"
    assert handle.aclose_count == 1


def test_arun_agent_closes_when_run_raises_non_transient():
    handle = _FakeHandle()

    async def run():
        raise _HTTPErr(400)

    with pytest.raises(_HTTPErr):
        asyncio.run(arun_agent(handle, run, label="t", sleep=_anoop_sleep))
    assert handle.aclose_count == 1


def test_arun_agent_retries_transient_then_succeeds():
    handle = _FakeHandle()
    calls = {"n": 0}

    async def run():
        calls["n"] += 1
        if calls["n"] < 3:
            raise _HTTPErr(503)
        return "ok"

    out = asyncio.run(arun_agent(handle, run, label="t", sleep=_anoop_sleep))
    assert out == "ok"
    assert calls["n"] == 3
    assert handle.aclose_count == 1


def test_arun_agent_uses_fallback_after_primary_exhausted():
    handle = _FakeHandle()
    calls = {"primary": 0, "fallback": 0}

    async def primary():
        calls["primary"] += 1
        raise _HTTPErr(503)

    async def fallback():
        calls["fallback"] += 1
        return "fallback-ok"

    out = asyncio.run(
        arun_agent(handle, primary, label="t", fallback=fallback, sleep=_anoop_sleep)
    )
    assert out == "fallback-ok"
    assert calls["primary"] > 1
    assert calls["fallback"] == 1
    assert handle.aclose_count == 1


def test_arun_agent_offline_trace_input_output_noop():
    handle = _FakeHandle()

    async def run():
        return "done"

    out = asyncio.run(
        arun_agent(
            handle,
            run,
            label="offline-trace",
            session_id="s",
            project="pk-x",
            trace_input={"a": 1},
            sleep=_anoop_sleep,
        )
    )
    assert out == "done"
    assert handle.aclose_count == 1


def test_run_agent_supports_run_sync_style_run():
    """The sync wrapper must execute *run* loop-free so run_sync-style
    callables (asyncio.run / run_until_complete inside) work."""
    handle = _FakeHandle()

    async def payload():
        return "ok"

    def run():
        return asyncio.run(payload())

    assert run_agent(handle, run, label="t", sleep=_noop_sleep) == "ok"
    assert handle.closed == 1

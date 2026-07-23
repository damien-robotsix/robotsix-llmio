"""Async retry tests — acall_with_retry parity, edge cases, and sync wrappers."""

from __future__ import annotations

import asyncio

import pytest
from pydantic_ai import UsageLimitExceeded

from robotsix_llmio.core.constants import TRANSIENT_RETRIES
from robotsix_llmio.core.retry import (
    acall_with_retry,
    acall_with_retry_and_fallback,
    call_with_retry,
    call_with_retry_and_fallback,
)


class _HTTPErr(Exception):
    def __init__(self, status_code):
        self.status_code = status_code


async def _anoop_sleep(_d: float) -> None:
    return None


def _noop_sleep(_d: float) -> None:
    return None


# --- async parity: acall_with_retry mirrors call_with_retry -----------------


def test_acall_with_retry_retries_then_succeeds():
    calls = {"n": 0}

    async def fn():
        calls["n"] += 1
        if calls["n"] < 3:
            raise _HTTPErr(503)
        return "ok"

    out = asyncio.run(acall_with_retry(fn, sleep=_anoop_sleep))
    assert out == "ok"
    assert calls["n"] == 3


def test_acall_with_retry_reraises_fatal_immediately():
    calls = {"n": 0}

    async def fn():
        calls["n"] += 1
        raise _HTTPErr(400)

    with pytest.raises(_HTTPErr):
        asyncio.run(acall_with_retry(fn, sleep=_anoop_sleep))
    assert calls["n"] == 1  # no retries on fatal


def test_acall_with_retry_uses_provider_transient_predicate():
    calls = {"n": 0}

    async def fn():
        calls["n"] += 1
        if calls["n"] < 2:
            raise ValueError("provider-specific transient")
        return "ok"

    out = asyncio.run(
        acall_with_retry(
            fn,
            sleep=_anoop_sleep,
            is_transient_fn=lambda e: isinstance(e, ValueError),
        )
    )
    assert out == "ok"
    assert calls["n"] == 2


def test_acall_with_retry_rate_limit_fallback_one_shot():
    """UsageLimitExceeded is never retried; the fallback is tried exactly once."""
    calls = {"primary": 0, "fallback": 0}

    async def primary():
        calls["primary"] += 1
        raise UsageLimitExceeded("cap")

    async def fallback():
        calls["fallback"] += 1
        return "fallback-ok"

    out = asyncio.run(
        acall_with_retry(primary, sleep=_anoop_sleep, fallback_fn=fallback)
    )
    assert out == "fallback-ok"
    assert calls == {"primary": 1, "fallback": 1}


def test_acall_with_retry_rate_limit_no_fallback_reraises():
    async def primary():
        raise UsageLimitExceeded("cap")

    with pytest.raises(UsageLimitExceeded):
        asyncio.run(acall_with_retry(primary, sleep=_anoop_sleep))


def test_afallback_only_after_primary_local_retries_exhausted():
    calls = {"primary": 0, "fallback": 0}

    async def primary():
        calls["primary"] += 1
        raise _HTTPErr(503)  # always transient → exhausts retries

    async def fallback():
        calls["fallback"] += 1
        return "fallback-ok"

    out = asyncio.run(
        acall_with_retry_and_fallback(primary, fallback, sleep=_anoop_sleep)
    )
    assert out == "fallback-ok"
    assert calls["primary"] > 1
    assert calls["fallback"] == 1


def test_afallback_not_used_when_primary_succeeds():
    calls = {"primary": 0, "fallback": 0}

    async def primary():
        calls["primary"] += 1
        return "primary-ok"

    async def fallback():
        calls["fallback"] += 1
        return "fallback-ok"

    out = asyncio.run(
        acall_with_retry_and_fallback(primary, fallback, sleep=_anoop_sleep)
    )
    assert out == "primary-ok"
    assert calls == {"primary": 1, "fallback": 0}


def test_aboth_fail_raises_fallback_chained_to_primary():
    class PrimaryErr(Exception):
        pass

    class FallbackErr(Exception):
        pass

    async def primary():
        raise PrimaryErr("primary")

    async def fallback():
        raise FallbackErr("fallback")

    with pytest.raises(FallbackErr) as ei:
        asyncio.run(
            acall_with_retry_and_fallback(primary, fallback, sleep=_anoop_sleep)
        )
    assert isinstance(ei.value.__cause__, PrimaryErr)


def test_ashould_fallback_false_reraises_primary_without_fallback():
    calls = {"fallback": 0}

    async def primary():
        raise ValueError("nope")

    async def fallback():
        calls["fallback"] += 1
        return "fallback-ok"

    with pytest.raises(ValueError):
        asyncio.run(
            acall_with_retry_and_fallback(
                primary,
                fallback,
                sleep=_anoop_sleep,
                should_fallback=lambda _e: False,
            )
        )
    assert calls["fallback"] == 0


# --- retries-exhausted edge case -------------------------------------------


def test_call_with_retry_retries_exhausted_reraises_last_error():
    """When every attempt raises a transient error, the last error re-raises."""
    calls = {"n": 0}

    def fn():
        calls["n"] += 1
        raise _HTTPErr(503)

    with pytest.raises(_HTTPErr):
        call_with_retry(fn, sleep=_noop_sleep)

    assert calls["n"] == TRANSIENT_RETRIES + 1


def test_acall_with_retry_retries_exhausted_reraises_last_error():
    """Async: when every attempt raises a transient error, the last error re-raises."""
    calls = {"n": 0}

    async def fn():
        calls["n"] += 1
        raise _HTTPErr(503)

    with pytest.raises(_HTTPErr):
        asyncio.run(acall_with_retry(fn, sleep=_anoop_sleep))

    assert calls["n"] == TRANSIENT_RETRIES + 1


# --- rate-limit on last attempt → fallback still fires (regression) --------


def test_call_with_retry_rate_limit_on_last_attempt_triggers_fallback():
    """Rate-limit error on the final attempt must still trigger the one-shot
    fallback — the ``attempt >= attempts`` guard must not short-circuit it."""
    calls = {"primary": 0, "fallback": 0}

    def primary():
        calls["primary"] += 1
        if calls["primary"] <= TRANSIENT_RETRIES:  # attempts 0..N-1
            raise _HTTPErr(503)  # transient, consumes a slot
        # Last attempt — rate limit
        raise UsageLimitExceeded("cap")

    def fallback():
        calls["fallback"] += 1
        return "fallback-ok"

    out = call_with_retry(primary, sleep=_noop_sleep, fallback_fn=fallback)
    assert out == "fallback-ok"
    assert calls["primary"] == TRANSIENT_RETRIES + 1
    assert calls["fallback"] == 1


def test_call_with_retry_fallback_gets_full_retry_budget():
    """Fallback activation must not consume a transient-retry slot — the
    fallback gets the same retry budget as the primary regardless of how many
    transient retries the primary consumed before hitting the rate limit."""
    calls = {"primary": 0, "fallback": 0}

    def primary():
        calls["primary"] += 1
        if calls["primary"] <= 3:  # burn 3 slots as transient
            raise _HTTPErr(503)
        raise UsageLimitExceeded("cap")

    def fallback():
        calls["fallback"] += 1
        if calls["fallback"] <= TRANSIENT_RETRIES:  # should be able to retry fully
            raise _HTTPErr(503)  # transient fallback error
        return "fallback-ok"

    out = call_with_retry(primary, sleep=_noop_sleep, fallback_fn=fallback)
    assert out == "fallback-ok"
    assert calls["primary"] == 4
    # Fallback got full retry budget: 1 initial + TRANSIENT_RETRIES retries = 5.
    assert calls["fallback"] == TRANSIENT_RETRIES + 1


def test_acall_with_retry_rate_limit_on_last_attempt_triggers_fallback():
    """Async: rate-limit error on the final attempt must still trigger the
    one-shot fallback."""
    calls = {"primary": 0, "fallback": 0}

    async def primary():
        calls["primary"] += 1
        if calls["primary"] <= TRANSIENT_RETRIES:  # attempts 0..N-1
            raise _HTTPErr(503)
        raise UsageLimitExceeded("cap")

    async def fallback():
        calls["fallback"] += 1
        return "fallback-ok"

    out = asyncio.run(
        acall_with_retry(primary, sleep=_anoop_sleep, fallback_fn=fallback)
    )
    assert out == "fallback-ok"
    assert calls["primary"] == TRANSIENT_RETRIES + 1
    assert calls["fallback"] == 1


def test_acall_with_retry_fallback_gets_full_retry_budget():
    """Async: fallback activation must not consume a transient-retry slot."""
    calls = {"primary": 0, "fallback": 0}

    async def primary():
        calls["primary"] += 1
        if calls["primary"] <= 3:
            raise _HTTPErr(503)
        raise UsageLimitExceeded("cap")

    async def fallback():
        calls["fallback"] += 1
        if calls["fallback"] <= TRANSIENT_RETRIES:
            raise _HTTPErr(503)
        return "fallback-ok"

    out = asyncio.run(
        acall_with_retry(primary, sleep=_anoop_sleep, fallback_fn=fallback)
    )
    assert out == "fallback-ok"
    assert calls["primary"] == 4
    assert calls["fallback"] == TRANSIENT_RETRIES + 1


# --- backoff jitter-after-cap regression -----------------------------------


def test_backoff_delay_never_exceeds_cap_sync(monkeypatch):
    """Jitter is applied to the raw exponential value BEFORE capping, so the
    final delay never exceeds TRANSIENT_BACKOFF_CAP regardless of jitter."""
    from robotsix_llmio.core.constants import TRANSIENT_BACKOFF_CAP, TRANSIENT_RETRIES

    # Force uniform(a, b) to always return b (max jitter).
    # _compute_backoff lives in core.retry; the random module reference
    # lives there.
    monkeypatch.setattr(
        "robotsix_llmio.core.retry.random.uniform",
        lambda a, b: b,
    )
    delays: list[float] = []

    def record_sleep(d: float) -> None:
        delays.append(d)

    calls = {"n": 0}

    def fn():
        calls["n"] += 1
        raise _HTTPErr(503)

    with pytest.raises(_HTTPErr):
        call_with_retry(fn, sleep=record_sleep)

    # One delay per retry attempt (TRANSIENT_RETRIES total).
    assert len(delays) == TRANSIENT_RETRIES
    for delay in delays:
        assert delay <= TRANSIENT_BACKOFF_CAP, (
            f"delay {delay} exceeds cap {TRANSIENT_BACKOFF_CAP}"
        )


def test_backoff_delay_never_exceeds_cap_async(monkeypatch):
    """Async: jitter-before-cap — delay never exceeds TRANSIENT_BACKOFF_CAP."""
    from robotsix_llmio.core.constants import TRANSIENT_BACKOFF_CAP, TRANSIENT_RETRIES

    monkeypatch.setattr(
        "robotsix_llmio.core.retry.random.uniform",
        lambda a, b: b,
    )
    delays: list[float] = []

    async def record_sleep(d: float) -> None:
        delays.append(d)

    calls = {"n": 0}

    async def fn():
        calls["n"] += 1
        raise _HTTPErr(503)

    with pytest.raises(_HTTPErr):
        asyncio.run(acall_with_retry(fn, sleep=record_sleep))

    assert len(delays) == TRANSIENT_RETRIES
    for delay in delays:
        assert delay <= TRANSIENT_BACKOFF_CAP, (
            f"delay {delay} exceeds cap {TRANSIENT_BACKOFF_CAP}"
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

"""Rate-limit fallback and backoff jitter regression tests."""

from __future__ import annotations

import asyncio

import pytest
from conftest import (
    _anoop_sleep,
    _HTTPErr,
    _noop_sleep,
)
from pydantic_ai import UsageLimitExceeded

from robotsix_llmio.core.constants import TRANSIENT_RETRIES
from robotsix_llmio.core.retry import (
    acall_with_retry,
    call_with_retry,
)

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

    # Force random.random to always return 1.0 (max jitter).  _compute_backoff
    # may live in robotsix_http.retry (when installed) or in the local fallback
    # — patch whichever is active.
    monkeypatch.setattr("robotsix_http.retry.random.random", lambda: 1.0)

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

    monkeypatch.setattr("robotsix_http.retry.random.random", lambda: 1.0)

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

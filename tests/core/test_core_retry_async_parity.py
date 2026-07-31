"""Async retry parity tests — acall_with_retry mirrors call_with_retry."""

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
    acall_with_retry_and_fallback,
    call_with_retry,
)

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

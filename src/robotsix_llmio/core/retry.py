"""Bounded retry + backoff for transient model/network failures and
rate-limit (``UsageLimitExceeded``) errors.

Transient = a ``ModelHTTPError`` with status 429 or 5xx, an httpx
timeout/transport error, an httpx 429/5xx response, a malformed-JSON decode,
or a provider-supplied extra signature (see ``is_transient_fn``). These ride
out with a short exponential backoff.

``UsageLimitExceeded`` is NOT transient (re-running immediately just re-hits
the cap) but IS rate-limited: handled with a longer schedule and an optional
one-shot fallback callable.

Everything else surfaces immediately, unchanged.

Parameters (retry counts, backoff) come from :mod:`.constants` — they are not
tunable per call.
"""

from __future__ import annotations

import asyncio
import dataclasses
import json
import logging
import random
import time
from collections.abc import Awaitable, Callable, Coroutine, Iterator
from typing import Any, TypeVar, cast

import httpx
from pydantic_ai import UsageLimitExceeded

# ---------------------------------------------------------------------------
# robotsix_http.retry — shared generic primitives, prefer their implementations
# when the package is installed (CI, proper dev env).  Fall back to local stubs
# when robotsix-http is unavailable (e.g. bare ``pytest`` without ``uv sync``).
# ---------------------------------------------------------------------------
try:
    from robotsix_http.retry import (
        RetryConfig,
        _compute_backoff,
        _status,
        _walk_cause_chain,
        is_transient,
    )
except ImportError:  # pragma: no cover — fallback path only
    # ------------------------------------------------------------------
    # Local fallback stubs — functionally equivalent to robotsix_http.retry
    # ------------------------------------------------------------------

    @dataclasses.dataclass(frozen=True)
    class RetryConfig:
        """Immutable retry configuration (fallback stub)."""

        max_retries: int = 4
        backoff_base: float = 2.0
        backoff_cap: float = 30.0
        jitter_factor: float = 0.5
        on_retry: Callable[[Exception, int, RetryConfig], None] | None = None

    def _compute_backoff(attempt: int, config: RetryConfig) -> float:
        delay: float = min(config.backoff_base**attempt, config.backoff_cap)
        jitter: float = delay * config.jitter_factor * random.random()
        return delay - jitter

    def _status(exc: BaseException) -> int | None:
        status: Any = getattr(exc, "status_code", None)
        if isinstance(status, int):
            return status
        response: Any = getattr(exc, "response", None)
        if response is not None:
            status = getattr(response, "status_code", None)
            if isinstance(status, int):
                return status
            status = getattr(response, "status", None)
            if isinstance(status, int):
                return status
        return None

    def _walk_cause_chain(
        exc: BaseException, max_depth: int = 10
    ) -> Iterator[BaseException]:
        current: BaseException | None = exc
        for _ in range(max_depth):
            if current is None:
                break
            yield current
            current = current.__cause__ or current.__context__

    def is_transient(exc: BaseException) -> bool:
        """Return ``True`` for exceptions that warrant a retry (fallback stub)."""
        if isinstance(
            exc,
            (
                httpx.TimeoutException,
                httpx.TransportError,
                json.JSONDecodeError,
            ),
        ):
            return True
        status = _status(exc)
        if status is not None and (status == 429 or 500 <= status < 600):
            return True
        for cause in _walk_cause_chain(exc):
            if cause is exc:
                continue
            if isinstance(
                cause,
                (
                    httpx.TimeoutException,
                    httpx.TransportError,
                    json.JSONDecodeError,
                ),
            ):
                return True
            s = _status(cause)
            if s is not None and (s == 429 or 500 <= s < 600):
                return True
        return False


from . import constants
from ._otel import get_recording_span
from .cost import flush_current_provider

log = logging.getLogger("robotsix_llmio.retry")

T = TypeVar("T")

__all__ = [
    "RetryConfig",
    "_compute_backoff",
    "_status",
    "_walk_cause_chain",
    "acall_with_retry",
    "acall_with_retry_and_fallback",
    "call_with_retry",
    "call_with_retry_and_fallback",
    "is_rate_limited",
    "is_transient",
]

# ---------------------------------------------------------------------------
# Loop-free sync coroutine driver (LLM-domain — load-bearing for run_sync
# callables).
#
# robotsix_http._drive_sync uses asyncio.run(), which creates a running event
# loop.  That breaks run_sync-style callables (pydantic-ai Agent.run_sync,
# claude_sdk tool agent) that themselves call asyncio.run() or
# loop.run_until_complete().  We keep a loop-free driver that advances the
# coroutine via send(None) — no event loop is ever running, so the wrapped
# callable is free to create its own.
# ---------------------------------------------------------------------------


def _drive_sync[R](coro: Coroutine[Any, Any, R]) -> R:
    """Drive *coro* to completion synchronously, without an event loop.

    The sync wrappers (:func:`call_with_retry`,
    :func:`~robotsix_llmio.core.tier_fallback.call_with_tier_fallback`,
    :func:`~robotsix_llmio.core.run.run_agent`) reuse the shared async loops
    with adapters that never suspend — a sync ``invoke`` and a blocking
    ``sleep`` — so the coroutine finishes on the first ``send``.

    Running it loop-free is load-bearing, not an optimisation: the wrapped
    ``fn`` is often ``run_sync``-style (``asyncio.run()`` or
    ``get_event_loop().run_until_complete()`` inside, e.g. pydantic-ai
    ``Agent.run_sync`` or the claude_sdk tool agent), which raises
    ``RuntimeError`` when invoked from a running event loop.  Wrapping the
    shared loop in ``asyncio.run()`` therefore broke every run_sync-style
    callable.
    """
    try:
        coro.send(None)
    except StopIteration as stop:
        return cast(R, stop.value)
    finally:
        coro.close()
    raise RuntimeError(
        "sync retry driver: coroutine unexpectedly suspended — a sync "
        "wrapper adapter awaited something asynchronous"
    )


# ---------------------------------------------------------------------------
# OTel span recording for transient retries, wired through
# ``robotsix_http.retry.RetryConfig.on_retry``
# ---------------------------------------------------------------------------


def _record_transient_retry_span(
    exc: Exception, attempt: int, config: RetryConfig
) -> None:
    """Record transient-retry metrics on the current OTel span; no-op without OTel.

    Signature matches ``robotsix_http.retry.RetryConfig.on_retry``:
    ``(exc, attempt, config)`` where *attempt* is 0-indexed.
    """
    span = get_recording_span()
    if span is None:
        return
    delay = _compute_backoff(attempt, config)
    span.set_attribute("llmio.retry.count", attempt + 1)
    span.set_attribute("llmio.retry.backoff_seconds", delay)
    span.set_attribute("llmio.retry.error", type(exc).__name__)


_RETRY_CONFIG = RetryConfig(
    max_retries=constants.TRANSIENT_RETRIES,
    backoff_base=constants.TRANSIENT_BACKOFF_BASE,
    backoff_cap=constants.TRANSIENT_BACKOFF_CAP,
    on_retry=_record_transient_retry_span,
)


# ---------------------------------------------------------------------------
# LLM-domain: rate-limit / UsageLimitExceeded helpers
# ---------------------------------------------------------------------------


def is_rate_limited(exc: BaseException) -> bool:
    """True only for ``UsageLimitExceeded`` (the pydantic-ai budget-cap
    exception). Walks the cause/context chain."""
    return any(isinstance(cur, UsageLimitExceeded) for cur in _walk_cause_chain(exc))


def _record_rate_limit_span(
    count: int,
    cumulative_backoff: float,
    fallback_activated: bool,
) -> None:
    """Record rate-limit metrics on the current OTel span; no-op without OTel."""
    span = get_recording_span()
    if span is None:
        return
    span.set_attribute("llmio.rate_limit.count", count)
    span.set_attribute("llmio.rate_limit.backoff_seconds", cumulative_backoff)
    if fallback_activated:
        span.set_attribute("llmio.rate_limit.fallback_activated", True)


def _handle_rate_limit(
    using_fallback: bool,
    fallback_fn: object | None,
    cumulative_backoff: float,
    what: str,
) -> None:
    """Handle a ``UsageLimitExceeded`` error.

    If a *fallback_fn* is available and not yet in use, activate it:
    log, record the span, and return so the caller resets the attempt
    counter and continues with the fallback.  Otherwise flush and
    re-raise immediately.
    """
    if not using_fallback and fallback_fn is not None:
        log.warning(
            "%s: rate-limit fallback activated on first UsageLimitExceeded",
            what,
        )
        _record_rate_limit_span(
            count=1,
            cumulative_backoff=cumulative_backoff,
            fallback_activated=True,
        )
        return
    _safe_flush()
    raise


def _handle_transient(
    e: Exception,
    attempt: int,
    config: RetryConfig,
    cumulative_backoff: float,
    what: str,
) -> tuple[float, float]:
    """Handle a transient error: compute backoff, log, flush.

    Returns ``(delay, new_cumulative_backoff)`` so the caller can sleep
    (sync or async) and increment *attempt*.  Raises if retries are
    exhausted.
    """
    if attempt >= config.max_retries:
        _safe_flush()
        raise
    delay = _compute_backoff(attempt, config)
    cumulative_backoff += delay
    log.warning(
        "%s: transient %s (attempt %d/%d) — retrying in %.1fs",
        what,
        type(e).__name__,
        attempt + 1,
        config.max_retries,
        delay,
    )
    _safe_flush()
    return delay, cumulative_backoff


# ---------------------------------------------------------------------------
# Core retry loop (async) — LLM-domain with rate-limit / fallback logic
# ---------------------------------------------------------------------------


async def _retry_loop(
    fn: Callable[[], Any],
    *,
    invoke: Callable[[Callable[[], Any]], Awaitable[Any]],
    sleep_fn: Callable[[float], Awaitable[None]],
    config: RetryConfig = _RETRY_CONFIG,
    what: str = "model call",
    fallback_fn: Callable[[], Any] | None = None,
    is_transient_fn: Callable[[BaseException], bool] = is_transient,
) -> Any:
    """Shared retry loop — ``invoke`` and ``sleep_fn`` adapt sync vs async.

    Transient retries use exponential backoff from *config*; span recording
    is wired through ``config.on_retry``.  ``UsageLimitExceeded`` triggers
    a one-shot fallback (if provided) rather than a retry.
    """
    using_fallback = False
    cumulative_backoff = 0.0

    attempt = 0
    while attempt <= config.max_retries:
        try:
            if using_fallback:
                assert fallback_fn is not None  # type-narrowing
                return await invoke(fallback_fn)
            return await invoke(fn)
        except Exception as e:
            if is_rate_limited(e):
                _handle_rate_limit(
                    using_fallback, fallback_fn, cumulative_backoff, what
                )
                using_fallback = True
                attempt = 0  # fresh retry budget for fallback
                continue

            if is_transient_fn(e):
                delay, cumulative_backoff = _handle_transient(
                    e, attempt, config, cumulative_backoff, what
                )
                if config.on_retry is not None:
                    config.on_retry(e, attempt, config)
                await sleep_fn(delay)
                attempt += 1
                continue

            # non-retryable
            _safe_flush()
            raise
    raise AssertionError("unreachable")  # pragma: no cover


# ---------------------------------------------------------------------------
# Public API (sync + async)
# ---------------------------------------------------------------------------


def call_with_retry[T](
    fn: Callable[[], T],
    *,
    what: str = "model call",
    sleep: Callable[[float], None] = time.sleep,
    fallback_fn: Callable[[], T] | None = None,
    is_transient_fn: Callable[[BaseException], bool] = is_transient,
) -> T:
    """Run ``fn`` and retry it on transient failures only.

    Transient failures use a short exponential backoff (baked base/cap).
    ``UsageLimitExceeded`` is never retried: if a ``fallback_fn`` is provided
    it is tried exactly once, else the exception re-raises immediately.
    Non-transient errors re-raise immediately; the last error re-raises once
    retries are exhausted.

    Fallback activation does **not** consume a transient-retry slot — the
    fallback gets the same full retry budget as the primary, regardless of how
    many transient retries the primary consumed before hitting the rate limit.

    *is_transient_fn* lets a provider layer widen the transient set (e.g. the
    OpenRouter upstream-error or DeepSeek reasoning-400 signatures).
    """

    async def _invoke(f: Callable[[], Any]) -> Any:
        return f()

    async def _sleep(d: float) -> None:
        sleep(d)

    return _drive_sync(  # type: ignore[no-any-return]
        _retry_loop(
            fn,
            invoke=_invoke,
            sleep_fn=_sleep,
            what=what,
            fallback_fn=fallback_fn,
            is_transient_fn=is_transient_fn,
        )
    )


def call_with_retry_and_fallback[T](
    primary: Callable[[], T],
    fallback: Callable[[], T] | None,
    *,
    what: str = "model call",
    sleep: Callable[[float], None] = time.sleep,
    is_transient_primary: Callable[[BaseException], bool] = is_transient,
    is_transient_fallback: Callable[[BaseException], bool] = is_transient,
    should_fallback: Callable[[BaseException], bool] = lambda _e: True,
) -> T:
    """Retry *primary* locally, then fall back to a different model.

    *primary* runs through :func:`call_with_retry` — its own bounded transient
    retries. Only when that whole session fails (retries exhausted, or a
    non-transient/terminal error) AND *fallback* is provided AND
    *should_fallback* of the error is true is *fallback* run, itself through a
    fresh :func:`call_with_retry` session.

    This is "retry locally first, fall back only when local retries failed": the
    fallback model is never tried *instead* of the primary's retries, only
    *after* they are exhausted. *is_transient_primary*/*is_transient_fallback*
    let each model retry on its own provider's transient signatures.
    """
    try:
        return call_with_retry(
            primary, what=what, sleep=sleep, is_transient_fn=is_transient_primary
        )
    except Exception as primary_exc:
        if fallback is None or not should_fallback(primary_exc):
            raise
        log.warning(
            "%s: primary failed after local retries (%s) — falling back to the "
            "secondary model",
            what,
            type(primary_exc).__name__,
        )
        _safe_flush()
        try:
            return call_with_retry(
                fallback,
                what=f"{what} (fallback)",
                sleep=sleep,
                is_transient_fn=is_transient_fallback,
            )
        except Exception as fallback_exc:
            # Chain the primary cause so the original failure isn't lost when the
            # fallback also fails.
            raise fallback_exc from primary_exc


async def acall_with_retry[T](
    fn: Callable[[], Awaitable[T]],
    *,
    what: str = "model call",
    sleep: Callable[[float], Awaitable[None]] = asyncio.sleep,
    fallback_fn: Callable[[], Awaitable[T]] | None = None,
    is_transient_fn: Callable[[BaseException], bool] = is_transient,
) -> T:
    """Async mirror of :func:`call_with_retry`.

    Identical retry/backoff/classification/one-shot-fallback semantics, except
    ``fn``/``fallback_fn`` are awaited and ``sleep`` defaults to
    :func:`asyncio.sleep` (kept injectable so tests can pass a no-op).
    """

    async def _invoke(f: Callable[[], Any]) -> Any:
        return await f()

    return await _retry_loop(  # type: ignore[no-any-return]
        fn,
        invoke=_invoke,
        sleep_fn=sleep,
        what=what,
        fallback_fn=fallback_fn,
        is_transient_fn=is_transient_fn,
    )


async def acall_with_retry_and_fallback[T](
    primary: Callable[[], Awaitable[T]],
    fallback: Callable[[], Awaitable[T]] | None,
    *,
    what: str = "model call",
    sleep: Callable[[float], Awaitable[None]] = asyncio.sleep,
    is_transient_primary: Callable[[BaseException], bool] = is_transient,
    is_transient_fallback: Callable[[BaseException], bool] = is_transient,
    should_fallback: Callable[[BaseException], bool] = lambda _e: True,
) -> T:
    """Async mirror of :func:`call_with_retry_and_fallback`.

    Retries *primary* locally through :func:`acall_with_retry`, then falls back
    to *fallback* only when that whole session fails AND *should_fallback* of
    the error is true. Same "retry locally first, fall back only when local
    retries failed" semantics, including ``raise fallback_exc from primary_exc``
    chaining when the fallback also fails.
    """
    try:
        return await acall_with_retry(
            primary, what=what, sleep=sleep, is_transient_fn=is_transient_primary
        )
    except Exception as primary_exc:
        if fallback is None or not should_fallback(primary_exc):
            raise
        log.warning(
            "%s: primary failed after local retries (%s) — falling back to the "
            "secondary model",
            what,
            type(primary_exc).__name__,
        )
        _safe_flush()
        try:
            return await acall_with_retry(
                fallback,
                what=f"{what} (fallback)",
                sleep=sleep,
                is_transient_fn=is_transient_fallback,
            )
        except Exception as fallback_exc:
            # Chain the primary cause so the original failure isn't lost when the
            # fallback also fails.
            raise fallback_exc from primary_exc


def _safe_flush() -> None:
    try:
        flush_current_provider()
    except Exception:
        log.warning("trace flush failed", exc_info=True)

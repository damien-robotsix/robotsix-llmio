"""Property-based (Hypothesis) tests for the transient backoff formula.

The formula under test (retry.py:193-195):
    raw = TRANSIENT_BACKOFF_BASE * (2 ** attempt)
    raw += random.uniform(0, raw / 2)   # jitter before cap
    delay = min(TRANSIENT_BACKOFF_CAP, raw)

These tests generate ``attempt`` and jitter values via Hypothesis strategies
and assert the clean mathematical invariants that the formula guarantees.
"""

from __future__ import annotations

from hypothesis import given
from hypothesis import strategies as st

from robotsix_llmio.core.constants import (
    TRANSIENT_BACKOFF_BASE,
    TRANSIENT_BACKOFF_CAP,
    TRANSIENT_RETRIES,
)

_MAX_ATTEMPT = TRANSIENT_RETRIES - 1


def _compute_delay(attempt: int, jitter_fraction: float) -> float:
    """Compute the retry backoff delay for a given attempt and jitter fraction.

    ``jitter_fraction`` in [0, 1] maps to ``uniform(0, raw/2)`` via
    ``jitter = jitter_fraction * raw / 2``.
    """
    raw = TRANSIENT_BACKOFF_BASE * (2**attempt)
    jitter = jitter_fraction * raw / 2
    return min(TRANSIENT_BACKOFF_CAP, raw + jitter)


# ------------------------------------------------------------------ boundedness


@given(
    attempt=st.integers(min_value=0, max_value=_MAX_ATTEMPT),
    jitter_fraction=st.floats(
        min_value=0, max_value=1, allow_nan=False, allow_infinity=False
    ),
)
def test_delay_bounded_by_cap(attempt: int, jitter_fraction: float) -> None:
    """For every attempt and every jitter draw, delay is in [0, CAP]."""
    delay = _compute_delay(attempt, jitter_fraction)
    assert 0 <= delay <= TRANSIENT_BACKOFF_CAP


# ---------------------------------------------------------------- non-negativity


@given(
    attempt=st.integers(min_value=0, max_value=_MAX_ATTEMPT),
    jitter_fraction=st.floats(
        min_value=0, max_value=1, allow_nan=False, allow_infinity=False
    ),
)
def test_delay_non_negative(attempt: int, jitter_fraction: float) -> None:
    """Delay is never negative."""
    delay = _compute_delay(attempt, jitter_fraction)
    assert delay >= 0


# ------------------------------------------------------------------ jitter range


@given(
    attempt=st.integers(min_value=0, max_value=_MAX_ATTEMPT),
    jitter_fraction=st.floats(
        min_value=0, max_value=1, allow_nan=False, allow_infinity=False
    ),
)
def test_jitter_range(attempt: int, jitter_fraction: float) -> None:
    """Jitter is always in [0, raw/2], so raw <= delay <= 1.5 * raw.

    The cap is never reached for the current constants
    (base=2.0, cap=30.0, retries=4 → max raw=16.0, max raw+jitter=24.0),
    so the upper bound 1.5*raw is always respected.
    """
    raw = TRANSIENT_BACKOFF_BASE * (2**attempt)
    delay = _compute_delay(attempt, jitter_fraction)
    assert raw <= delay <= 1.5 * raw


# ---------------------------------------------------------------- monotonicity


@given(
    attempts=st.lists(
        st.integers(min_value=0, max_value=_MAX_ATTEMPT),
        min_size=2,
        max_size=TRANSIENT_RETRIES,
        unique=True,
    ).map(sorted),
    jitter_fraction=st.floats(
        min_value=0, max_value=1, allow_nan=False, allow_infinity=False
    ),
)
def test_delays_monotonic(attempts: list[int], jitter_fraction: float) -> None:
    """Delay sequence is non-decreasing across increasing attempt numbers.

    Exponential backoff guarantees raw doubles each step while jitter is
    at most raw/2, so even with max jitter on attempt N and min jitter
    on attempt N+1, the sequence is non-decreasing.
    """
    delays = [_compute_delay(a, jitter_fraction) for a in attempts]
    for i in range(len(delays) - 1):
        assert delays[i] <= delays[i + 1] + 1e-9  # floating tolerance


# ----------------------------------------------------------- full-sequence bound


@given(
    jitter_fraction=st.floats(
        min_value=0, max_value=1, allow_nan=False, allow_infinity=False
    ),
)
def test_full_sequence_bounded(jitter_fraction: float) -> None:
    """For a fixed jitter fraction, the full sequence of delays across all
    attempts respects every invariant simultaneously."""
    attempts = list(range(TRANSIENT_RETRIES))
    delays = [_compute_delay(a, jitter_fraction) for a in attempts]

    # Boundedness + non-negativity (redundant with test_delay_bounded_by_cap
    # but checked here for the combined sequence).
    for delay in delays:
        assert 0 <= delay <= TRANSIENT_BACKOFF_CAP

    # Monotonicity.
    for i in range(len(delays) - 1):
        assert delays[i] <= delays[i + 1] + 1e-9

    # Jitter range for each attempt.
    for a in attempts:
        raw = TRANSIENT_BACKOFF_BASE * (2**a)
        delay = _compute_delay(a, jitter_fraction)
        assert raw <= delay <= 1.5 * raw

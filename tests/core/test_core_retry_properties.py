"""Property-based tests for the retry backoff formula.

These tests verify invariants of the exponential backoff with jitter
used in `src/robotsix_llmio/core/retry.py`.
"""

from hypothesis import given
from hypothesis import strategies as st

from robotsix_llmio.core.constants import (
    TRANSIENT_BACKOFF_BASE,
    TRANSIENT_BACKOFF_CAP,
    TRANSIENT_RETRIES,
)


def _compute_delay(attempt: int, jitter_fraction: float) -> float:
    """Compute backoff delay for a given attempt and jitter fraction.

    `jitter_fraction` in [0, 0.5] — the proportion of the pre-jitter raw
    value added as jitter.  The real code uses ``random.uniform(0, raw/2)``,
    equivalent to ``raw * uniform(0, 0.5)``.
    """
    raw = TRANSIENT_BACKOFF_BASE * (2**attempt)
    raw += raw * jitter_fraction
    return min(TRANSIENT_BACKOFF_CAP, raw)


@given(
    attempt=st.integers(min_value=0, max_value=TRANSIENT_RETRIES - 1),
    jitter=st.floats(
        min_value=0.0, max_value=0.5, allow_nan=False, allow_infinity=False
    ),
)
def test_delay_bounded_by_cap(attempt: int, jitter: float) -> None:
    """Delay never exceeds TRANSIENT_BACKOFF_CAP."""
    delay = _compute_delay(attempt, jitter)
    assert 0 <= delay <= TRANSIENT_BACKOFF_CAP


@given(
    attempt=st.integers(min_value=0, max_value=TRANSIENT_RETRIES - 1),
    jitter=st.floats(
        min_value=0.0, max_value=0.5, allow_nan=False, allow_infinity=False
    ),
)
def test_delay_non_negative(attempt: int, jitter: float) -> None:
    """Delay is always non-negative regardless of attempt or jitter."""
    delay = _compute_delay(attempt, jitter)
    assert delay >= 0


@given(
    jitters=st.lists(
        st.floats(min_value=0.0, max_value=0.5, allow_nan=False, allow_infinity=False),
        min_size=TRANSIENT_RETRIES,
        max_size=TRANSIENT_RETRIES,
    ),
)
def test_delays_monotonic(jitters: list[float]) -> None:
    """Delay sequence is non-decreasing across retry attempts for all
    jitter combinations."""
    delays = [_compute_delay(a, j) for a, j in enumerate(jitters)]
    for i in range(len(delays) - 1):
        assert delays[i] <= delays[i + 1] + 1e-9


@given(
    attempt=st.integers(min_value=0, max_value=TRANSIENT_RETRIES - 1),
    jitter=st.floats(
        min_value=0.0, max_value=0.5, allow_nan=False, allow_infinity=False
    ),
)
def test_jitter_range(attempt: int, jitter: float) -> None:
    """When not capped, delay is in [raw, 1.5*raw]."""
    raw_before = TRANSIENT_BACKOFF_BASE * (2**attempt)
    delay = _compute_delay(attempt, jitter)
    if delay < TRANSIENT_BACKOFF_CAP:
        assert raw_before <= delay <= 1.5 * raw_before

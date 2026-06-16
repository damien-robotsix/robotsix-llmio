"""Week-window math for the pace governor — pure functions, no I/O.

Anchor: day-of-week (0=Monday..6=Sunday) + time-of-day (HH:MM, UTC).
From a given anchor and a ``now``, compute the current week's [start, end)
window and the fraction elapsed.
"""

from __future__ import annotations

from datetime import datetime, timedelta


def _parse_anchor_time(anchor_time: str) -> tuple[int, int]:
    """Parse ``HH:MM`` into ``(hour, minute)``."""
    parts = anchor_time.split(":")
    if len(parts) != 2:
        raise ValueError(f"Invalid anchor time format: {anchor_time!r}")
    return int(parts[0]), int(parts[1])


def _current_week_window(
    now: datetime,
    anchor_day: int,
    anchor_time: str,
) -> tuple[datetime, datetime]:
    """Return ``(week_start, week_end)`` for the week containing *now*.

    The week starts at the most recent *anchor_day* + *anchor_time* ≤ *now*,
    and ends exactly 7 days later (exclusive bound for CostWindow).
    """
    hour, minute = _parse_anchor_time(anchor_time)
    # Floor *now* to the anchor time-of-day on the same calendar date.
    anchor_today = now.replace(hour=hour, minute=minute, second=0, microsecond=0)
    # Days since the anchor weekday (0=Monday).
    days_since_anchor = (now.weekday() - anchor_day) % 7
    week_start = anchor_today - timedelta(days=days_since_anchor)
    # If the anchor time today is still in the future, step back one full week.
    if week_start > now:
        week_start -= timedelta(days=7)
    week_end = week_start + timedelta(days=7)
    return week_start, week_end


def week_fraction_elapsed(
    now: datetime,
    week_start: datetime,
    week_end: datetime,
) -> float:
    """Fraction of the week elapsed at *now*, clamped to [0, 1]."""
    total = (week_end - week_start).total_seconds()
    if total <= 0:
        return 0.0
    elapsed = (now - week_start).total_seconds()
    if elapsed <= 0:
        return 0.0
    if elapsed >= total:
        return 1.0
    return elapsed / total

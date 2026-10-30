"""Reset-hint parsing — pure helpers for provider quota-reset text.

Extracted from :mod:`robotsix_llmio.core.failover`: stateless regex and
date arithmetic that turns an exhaustion message's "resets …" hint into a
delay in seconds. The failover tracker uses it to arm a window until the
named reset instead of the fixed ``window_seconds``.
"""

from __future__ import annotations

import re
from datetime import datetime, timedelta

# The Claude CLI names the quota reset inside the exhaustion text, in two
# shapes: the rolling cap ("resets 1:10pm (UTC)", same day or tomorrow) and
# the weekly cap ("resets Sep 5, 7pm (UTC)"). When a hint parses, the
# failover window arms until that time instead of the fixed
# ``window_seconds`` — probing a provider that TOLD us when it comes back
# just burns a failed CLI spawn every window and makes status UIs blink.
_RESET_HINT_RE = re.compile(
    r"resets\s+(?:(?P<month>[a-z]{3,9})\s+(?P<day>\d{1,2}),?\s+)?"
    r"(?P<hour>\d{1,2})(?::(?P<minute>\d{2}))?\s*(?P<meridiem>am|pm)\s*\(utc\)",
    re.IGNORECASE,
)

_MONTHS = {
    m: i
    for i, names in enumerate(
        (
            ("jan", "january"),
            ("feb", "february"),
            ("mar", "march"),
            ("apr", "april"),
            ("may",),
            ("jun", "june"),
            ("jul", "july"),
            ("aug", "august"),
            ("sep", "september"),
            ("oct", "october"),
            ("nov", "november"),
            ("dec", "december"),
        ),
        start=1,
    )
    for m in names
}

#: Defensive ceiling on a hint-derived window: a mis-parsed or hostile
#: "resets" value can never park the fleet on the paid slot for more than a
#: worst-case weekly cycle (+ slack).
_MAX_RESET_WINDOW_SECONDS: float = 8 * 24 * 3600.0

#: Armed windows end this long AFTER the hinted reset, so the first probe
#: lands when the quota is actually back rather than seconds before.
_RESET_SLACK_SECONDS: float = 120.0


def _parse_reset_delay(text: str, wall_now: datetime) -> float | None:
    """Seconds from *wall_now* until the reset named in *text*, or ``None``.

    Handles both CLI shapes: time-only rolls forward to the next occurrence
    (today or tomorrow); a month+day form pins that calendar date (rolling to
    next year when it already passed, e.g. "resets Jan 2" seen in late Dec).
    All times are interpreted as UTC per the ``(UTC)`` marker.
    """
    match = _RESET_HINT_RE.search(text)
    if match is None:
        return None

    hour = int(match.group("hour"))
    minute = int(match.group("minute") or 0)
    if not (1 <= hour <= 12) or minute > 59:
        return None
    hour24 = hour % 12
    if match.group("meridiem").lower() == "pm":
        hour24 += 12

    month_name = match.group("month")
    if month_name is not None:
        month = _MONTHS.get(month_name.lower())
        day = int(match.group("day"))
        if month is None:
            return None
        try:
            target = wall_now.replace(
                month=month,
                day=day,
                hour=hour24,
                minute=minute,
                second=0,
                microsecond=0,
            )
        except ValueError:
            return None
        if target <= wall_now:
            target = target.replace(year=target.year + 1)
    else:
        target = wall_now.replace(hour=hour24, minute=minute, second=0, microsecond=0)
        if target <= wall_now:
            target += timedelta(days=1)

    return (target - wall_now).total_seconds()

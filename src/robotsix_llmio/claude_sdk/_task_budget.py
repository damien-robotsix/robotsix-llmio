"""Claude Agent SDK ``task_budget`` construction.

This transport maps a tier's ``max_tokens`` onto ``ClaudeAgentOptions.task_budget``,
but the two are **not** the same control:

* ``max_tokens`` is a hard per-response output cap the model never sees.
* ``task_budget.total`` is an *advisory* budget for the whole agent loop
  (thinking + tool results + output). The server injects a countdown the model
  reads, so it paces itself and wraps up gracefully instead of being cut off.

The API enforces a floor on the advisory budget, so a ``max_tokens`` below it
cannot be expressed as a ``task_budget`` at all. Passing one through verbatim
makes the API reject **every** call for that agent, which is what this module
exists to prevent.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

if TYPE_CHECKING:  # type-only — keeps this module importable without the SDK
    from claude_agent_sdk import TaskBudget

logger = logging.getLogger(__name__)

# The Messages API rejects a lower ``task_budget.total`` outright:
#   400 `task_budget.total` must be at least 20,000 tokens for this model.
# A hard request-validation floor, not a per-model tunable.
TASK_BUDGET_MIN_TOTAL = 20_000

# Labels already warned about — the value comes from static config, so one
# warning per agent is the signal; repeating it on every call is noise.
# Bounded by the number of configured agents.
_clamp_warned: set[str] = set()


def build_task_budget(max_tokens: int | None, label: str) -> TaskBudget | None:
    """Build the ``task_budget`` option for *label* from a tier's *max_tokens*.

    ``None`` (no configured cap) yields ``None`` — an unbounded loop. A value at
    or above :data:`TASK_BUDGET_MIN_TOTAL` is passed through. A value *below* the
    floor is clamped up to it, warning once per *label*: the request would
    otherwise be rejected outright, and a working call with a larger-than-asked
    advisory budget beats an agent that cannot run at all. The operator's
    intent is not silently honoured — hence the warning.
    """
    if max_tokens is None:
        return None
    if max_tokens >= TASK_BUDGET_MIN_TOTAL:
        return {"total": max_tokens}
    if label not in _clamp_warned:
        _clamp_warned.add(label)
        logger.warning(
            "%s: configured max_tokens=%d is below the Claude Agent SDK "
            "task_budget floor of %d; clamping. The agent loop's advisory "
            "budget will be %d, not %d — lower max_tokens cannot be expressed "
            "as a task_budget. Set max_tokens >= %d to silence this.",
            label,
            max_tokens,
            TASK_BUDGET_MIN_TOTAL,
            TASK_BUDGET_MIN_TOTAL,
            max_tokens,
            TASK_BUDGET_MIN_TOTAL,
        )
    return {"total": TASK_BUDGET_MIN_TOTAL}

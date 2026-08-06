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

There is a second, independent precondition: ``task_budget`` is a beta
parameter only a subset of models accept. Every other model rejects the request
with ``400 This model does not support user-configurable task budgets``, which
— like the floor — kills **every** call for that agent. The supported set can
not be hardcoded here: callers configure a tier alias (``"sonnet"``,
``"opus"``), and the ``claude`` CLI resolves it to a concrete model downstream,
so this transport genuinely does not know what it is talking to. Instead the
rejection is treated as the discovery mechanism — :func:`run_with_task_budget`
drops the budget, retries once, and remembers the model so later calls skip
straight to the working request.
"""

from __future__ import annotations

import logging
from collections.abc import Awaitable, Callable
from dataclasses import replace
from typing import TYPE_CHECKING, Any

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

# Models observed rejecting ``task_budget`` outright. Populated from the API's
# own rejection rather than a hardcoded list, because the supported set moves
# with each model release and the tier alias is resolved downstream of this
# process (see the module docstring). Keyed by the SDK model string as
# configured, so an alias and a pinned ID are tracked independently — they can
# genuinely resolve differently. Bounded by the number of configured models.
#
# Process-local: a restart re-learns on the first call, at the cost of one
# extra round trip per model. That is the right trade against persisting it —
# the answer changes when Anthropic ships a model, and a stale "unsupported"
# on disk would silently withhold the budget long after it started working.
_unsupported_models: set[str] = set()


def mark_task_budget_unsupported(model: str, label: str) -> None:
    """Record that *model* rejects ``task_budget``, warning once.

    Called from :func:`run_with_task_budget` when the API says so. Subsequent
    :func:`build_task_budget` calls for *model* return ``None``, so the agent
    loop runs unbudgeted instead of unrunnable.
    """
    if model in _unsupported_models:
        return
    _unsupported_models.add(model)
    logger.warning(
        "%s: model %r rejected task_budget (the API reports it does not "
        "support user-configurable task budgets); retrying without it and "
        "omitting it for this model from now on. The agent loop runs with no "
        "advisory budget — max_tokens still caps each response.",
        label,
        model,
    )


def supports_task_budget(model: str) -> bool:
    """False once *model* has been seen rejecting ``task_budget``."""
    return model not in _unsupported_models


async def run_with_task_budget[T](
    run: Callable[[Any], Awaitable[T]],
    options: Any,  # ClaudeAgentOptions — untyped to avoid importing the SDK
    model: str,
    label: str,
) -> T:
    """Await ``run(options)``, retrying once without ``task_budget`` if the API
    rejects the parameter as unsupported for *model*.

    This is the discovery path for a fact the transport cannot know up front.
    Only that one specific ``400`` is retried — every other request-validation
    rejection still propagates, because those are genuinely reproducible and a
    retry would be pure waste. The retry re-sends the identical request minus
    the budget, so it is safe: ``task_budget`` is advisory and removing it
    changes pacing, not semantics.
    """
    from .transient import is_task_budget_unsupported_error

    try:
        return await run(options)
    except Exception as exc:
        if getattr(options, "task_budget", None) is None:
            raise
        if not is_task_budget_unsupported_error(exc):
            raise
        mark_task_budget_unsupported(model, label)
        return await run(replace(options, task_budget=None))


def build_task_budget(
    max_tokens: int | None, label: str, model: str | None = None
) -> TaskBudget | None:
    """Build the ``task_budget`` option for *label* from a tier's *max_tokens*.

    ``None`` (no configured cap) yields ``None`` — an unbounded loop. A value at
    or above :data:`TASK_BUDGET_MIN_TOTAL` is passed through. A value *below* the
    floor is clamped up to it, warning once per *label*: the request would
    otherwise be rejected outright, and a working call with a larger-than-asked
    advisory budget beats an agent that cannot run at all. The operator's
    intent is not silently honoured — hence the warning.

    *model*, when given, is checked against the set of models already observed
    rejecting the parameter (see :func:`mark_task_budget_unsupported`); a known
    rejecter yields ``None`` so the call goes out without a budget instead of
    re-earning the same ``400``.
    """
    if model is not None and not supports_task_budget(model):
        return None
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

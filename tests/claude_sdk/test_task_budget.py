"""Tests for the Claude Agent SDK task_budget floor."""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass

import pytest

from robotsix_llmio.claude_sdk._task_budget import (
    TASK_BUDGET_MIN_TOTAL,
    build_task_budget,
    run_with_task_budget,
    supports_task_budget,
)


@pytest.fixture(autouse=True)
def _reset_warn_state():
    from robotsix_llmio.claude_sdk import _task_budget

    _task_budget._clamp_warned.clear()
    _task_budget._unsupported_models.clear()
    yield
    _task_budget._clamp_warned.clear()
    _task_budget._unsupported_models.clear()


def test_none_max_tokens_yields_no_budget():
    assert build_task_budget(None, "agent") is None


def test_value_at_or_above_floor_passes_through():
    assert build_task_budget(TASK_BUDGET_MIN_TOTAL, "agent") == {
        "total": TASK_BUDGET_MIN_TOTAL
    }
    assert build_task_budget(65536, "agent") == {"total": 65536}


def test_value_below_floor_sends_no_budget(caplog):
    """Below the floor there is no honest budget to send.

    Passing it verbatim is a 400. Clamping UP is worse than either reading of
    the operator's intent: task_budget is advisory, so the clamp caps nothing
    and instead tells the model it has a 20,000-token allowance for the whole
    task — which is how agents ended up wrapping up before starting work.
    """
    with caplog.at_level(logging.WARNING):
        assert build_task_budget(8192, "refine") is None
    assert "below the Claude Agent SDK task_budget floor" in caplog.text
    assert "refine" in caplog.text


def test_clamp_warning_is_emitted_once_per_label(caplog):
    """The value is static config — warn once, don't flood the log per call."""
    with caplog.at_level(logging.WARNING):
        build_task_budget(8192, "refine")
        build_task_budget(8192, "refine")
        build_task_budget(8192, "retrospect")
    assert caplog.text.count("below the Claude Agent SDK task_budget floor") == 2


def test_floor_matches_documented_api_minimum():
    assert TASK_BUDGET_MIN_TOTAL == 20_000


# ---------------------------------------------------------------------------
# Unsupported-model discovery
#
# ``task_budget`` is a beta parameter only some models accept, and the tier
# alias is resolved by the CLI downstream of this process — so the transport
# learns the answer from the API's rejection rather than a hardcoded list.
#
# These drive the coroutine with ``asyncio.run`` rather than an async test
# plugin: this suite has no async tests and no asyncio plugin configured, and
# one helper is not worth a new test dependency.
# ---------------------------------------------------------------------------

_UNSUPPORTED_MSG = (
    "Anthropic API rejected the request (claude:sonnet): API Error: 400 "
    "This model does not support user-configurable task budgets."
)


@dataclass
class _Opts:
    """Stand-in for ``ClaudeAgentOptions`` — a mutable dataclass carrying the
    one field under test, so these tests don't need the SDK installed.
    ``dataclasses.replace`` (used by the helper) requires a real dataclass."""

    task_budget: dict | None = None
    model: str = "sonnet"


def test_unsupported_model_retries_once_without_budget():
    """The one 400 worth retrying: drop the budget, re-send, succeed."""
    seen: list[dict | None] = []

    async def run(opts):
        seen.append(opts.task_budget)
        if opts.task_budget is not None:
            raise RuntimeError(_UNSUPPORTED_MSG)
        return "ok"

    result = asyncio.run(
        run_with_task_budget(
            run, _Opts(task_budget={"total": 20_000}), "sonnet", "claude:sonnet"
        )
    )

    assert result == "ok"
    assert seen == [{"total": 20_000}, None], "expected one retry, budget dropped"
    assert not supports_task_budget("sonnet"), "model should be remembered"


def test_subsequent_calls_skip_the_budget_entirely():
    """After discovery, build_task_budget stops asking — no second round trip."""

    async def run(opts):
        if opts.task_budget is not None:
            raise RuntimeError(_UNSUPPORTED_MSG)
        return "ok"

    asyncio.run(
        run_with_task_budget(
            run, _Opts(task_budget={"total": 20_000}), "sonnet", "claude:sonnet"
        )
    )

    assert build_task_budget(50_000, "claude:sonnet", "sonnet") is None
    # A different model is unaffected — aliases can resolve differently.
    assert build_task_budget(50_000, "claude:opus", "opus") == {"total": 50_000}


def test_other_400s_are_not_retried():
    """Every other request-validation rejection is reproducible — retrying it
    would burn a round trip for nothing."""
    calls = 0

    async def run(opts):
        nonlocal calls
        calls += 1
        raise RuntimeError("API Error: 400 `task_budget.total` must be at least 20,000")

    with pytest.raises(RuntimeError):
        asyncio.run(
            run_with_task_budget(
                run, _Opts(task_budget={"total": 100}), "sonnet", "claude:sonnet"
            )
        )

    assert calls == 1, "a non-task-budget 400 must not be retried"
    assert supports_task_budget("sonnet")


def test_no_budget_means_no_retry():
    """With no budget to drop there is nothing to retry — don't loop."""
    calls = 0

    async def run(opts):
        nonlocal calls
        calls += 1
        raise RuntimeError(_UNSUPPORTED_MSG)

    with pytest.raises(RuntimeError):
        asyncio.run(
            run_with_task_budget(
                run, _Opts(task_budget=None), "sonnet", "claude:sonnet"
            )
        )

    assert calls == 1


def test_retry_failure_propagates():
    """If the budget-free retry also fails, the caller sees that error."""

    async def run(opts):
        raise RuntimeError(_UNSUPPORTED_MSG if opts.task_budget else "boom")

    with pytest.raises(RuntimeError, match="boom"):
        asyncio.run(
            run_with_task_budget(
                run, _Opts(task_budget={"total": 20_000}), "sonnet", "claude:sonnet"
            )
        )


def test_model_arg_is_optional_and_backward_compatible():
    """Existing two-arg callers keep working."""
    assert build_task_budget(50_000, "agent") == {"total": 50_000}

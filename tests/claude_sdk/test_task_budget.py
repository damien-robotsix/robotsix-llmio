"""Tests for the Claude Agent SDK task_budget floor."""

from __future__ import annotations

import logging

import pytest

from robotsix_llmio.claude_sdk._task_budget import (
    TASK_BUDGET_MIN_TOTAL,
    build_task_budget,
)


@pytest.fixture(autouse=True)
def _reset_warn_state():
    from robotsix_llmio.claude_sdk import _task_budget

    _task_budget._clamp_warned.clear()
    yield
    _task_budget._clamp_warned.clear()


def test_none_max_tokens_yields_no_budget():
    assert build_task_budget(None, "agent") is None


def test_value_at_or_above_floor_passes_through():
    assert build_task_budget(TASK_BUDGET_MIN_TOTAL, "agent") == {
        "total": TASK_BUDGET_MIN_TOTAL
    }
    assert build_task_budget(65536, "agent") == {"total": 65536}


def test_value_below_floor_is_clamped_not_passed_through(caplog):
    """The live outage: refine's max_tokens=8192 went through verbatim and the
    API rejected every single call with a 400."""
    with caplog.at_level(logging.WARNING):
        assert build_task_budget(8192, "refine") == {"total": TASK_BUDGET_MIN_TOTAL}
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

"""Unit tests for robotsix_llmio.tools built-in functions."""

from __future__ import annotations

import time

import pytest

from robotsix_llmio.tools import get_builtin_tools
from robotsix_llmio.tools._builtins import (
    calculator,
    echo,
    get_time,
    roll_dice,
)

# ---------------------------------------------------------------------------
# get_time
# ---------------------------------------------------------------------------


def test_get_time_returns_non_empty_iso_string():
    result = get_time()
    assert isinstance(result, str)
    assert len(result) > 0
    assert "T" in result


def test_get_time_rapid_calls_yield_different_timestamps():
    a = get_time()
    time.sleep(0.002)
    b = get_time()
    assert a != b


# ---------------------------------------------------------------------------
# echo
# ---------------------------------------------------------------------------


def test_echo_returns_text_unchanged():
    assert echo("hello") == "hello"


def test_echo_handles_empty_string():
    assert echo("") == ""


def test_echo_preserves_special_characters():
    text = "line1\nline2\tindented"
    assert echo(text) == text


# ---------------------------------------------------------------------------
# calculator — valid expressions
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "expression, expected",
    [
        ("2 + 3", "5"),
        ("10 - 7", "3"),
        ("4 * 5", "20"),
        ("10 / 3", "3.3333333333333335"),
        ("2 ** 10", "1024"),
        ("(2 + 3) * 4", "20"),
        ("-5 + 3", "-2"),
        ("3.5 + 2.5", "6"),
        ("0.1 + 0.2", "0.30000000000000004"),
    ],
)
def test_calculator_valid_expressions(expression: str, expected: str):
    assert calculator(expression) == expected


def test_calculator_nested_parentheses():
    assert calculator("((2 + 3) * (4 - 1))") == "15"


def test_calculator_unary_minus_literal():
    assert calculator("-42") == "-42"


def test_calculator_float_result_rounded_to_int_representation():
    """4 / 2 = 2.0 should be rendered as '2', not '2.0'."""
    assert calculator("4 / 2") == "2"


def test_calculator_large_power():
    assert calculator("10 ** 100").startswith("1")  # huge, but starts with 1


# ---------------------------------------------------------------------------
# calculator — error cases
# ---------------------------------------------------------------------------


def test_calculator_division_by_zero():
    result = calculator("10 / 0")
    assert result.startswith("Error: ")


def test_calculator_empty_expression_is_rejected():
    result = calculator("")
    assert result.startswith("Error: ")


def test_calculator_import_statement_is_rejected():
    """``import os`` is a statement, not an expression — rejected at parse time."""
    result = calculator("import os")
    assert result.startswith("Error: ")


def test_calculator_function_call_is_rejected():
    result = calculator("__import__('os')")
    assert result.startswith("Error: ")


def test_calculator_attribute_access_is_rejected():
    result = calculator("().__class__")
    assert result.startswith("Error: ")


def test_calculator_assignment_is_rejected():
    result = calculator("x = 1")
    assert result.startswith("Error: ")


def test_calculator_unsupported_operator_is_rejected():
    """``<<`` (bit shift) is not in the allowed operator set."""
    result = calculator("1 << 2")
    assert result.startswith("Error: ")


# ---------------------------------------------------------------------------
# roll_dice
# ---------------------------------------------------------------------------


def test_roll_dice_default_six_sided():
    for _ in range(100):
        result = roll_dice()
        assert isinstance(result, str)
        val = int(result)
        assert 1 <= val <= 6


def test_roll_dice_twenty_sided():
    for _ in range(100):
        result = roll_dice(20)
        val = int(result)
        assert 1 <= val <= 20


def test_roll_dice_thousand_sided_smoke():
    for _ in range(100):
        result = roll_dice(1000)
        val = int(result)
        assert 1 <= val <= 1000


def test_roll_dice_zero_sides_raises_valueerror():
    with pytest.raises(ValueError):
        roll_dice(0)


def test_roll_dice_negative_sides_raises_valueerror():
    with pytest.raises(ValueError):
        roll_dice(-1)


# ---------------------------------------------------------------------------
# get_builtin_tools
# ---------------------------------------------------------------------------


def test_get_builtin_tools_returns_list_of_four_callables():
    tools = get_builtin_tools()
    assert isinstance(tools, list)
    assert len(tools) == 4
    for tool in tools:
        assert callable(tool)


def test_get_builtin_tools_fresh_list_each_call():
    a = get_builtin_tools()
    b = get_builtin_tools()
    assert a is not b
    assert a == b


def test_get_builtin_tools_contains_expected_functions():
    tools = get_builtin_tools()
    assert get_time in tools
    assert echo in tools
    assert calculator in tools
    assert roll_dice in tools


def test_get_builtin_tools_compatible_with_build_agent():
    """Tools returned by ``get_builtin_tools()`` are plain callables
    with ``__name__`` and ``__doc__`` — exactly the contract that
    pydantic-ai's ``Agent(tools=...)`` relies on for introspection."""
    tools = get_builtin_tools()
    assert isinstance(tools, list)
    assert len(tools) == 4

    for tool in tools:
        assert callable(tool)
        # pydantic-ai discovers the tool name from __name__
        assert hasattr(tool, "__name__"), f"{tool!r} missing __name__"
        # and the description from __doc__
        assert tool.__doc__ is not None, f"{tool!r} missing docstring"

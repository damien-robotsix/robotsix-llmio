"""Built-in example tools for robotsix-llmio agents.

Four simple synchronous tools — ``get_time``, ``echo``, ``calculator``,
``roll_dice`` — plus a convenience getter ``get_builtin_tools`` that returns
them as a list ready for ``build_agent(tools=...)``.
"""

from __future__ import annotations

import ast
import operator
import random
from datetime import UTC, datetime

# ---------------------------------------------------------------------------
# Allowed operator tables for the safe calculator
# ---------------------------------------------------------------------------
from typing import Any

_BIN_OPS: dict[type, Any] = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.Pow: operator.pow,
}

_UNARY_OPS: dict[type, Any] = {
    ast.USub: operator.neg,
}


def _safe_eval(node: ast.AST) -> float | int:
    """Recursively evaluate a validated AST node using only allowed operations."""
    match node:
        case ast.Expression(body=body):
            return _safe_eval(body)
        case ast.Constant(value=int() | float()):
            return node.value  # type: ignore[return-value]
        case ast.BinOp(left=left, op=op, right=right):
            bin_op = type(op)
            if bin_op not in _BIN_OPS:
                raise ValueError(f"Unsupported operator: {bin_op.__name__}")
            left_val = _safe_eval(left)
            right_val = _safe_eval(right)
            return _BIN_OPS[bin_op](left_val, right_val)  # type: ignore[no-any-return]
        case ast.UnaryOp(op=op, operand=operand):
            unary_op = type(op)
            if unary_op not in _UNARY_OPS:
                raise ValueError(f"Unsupported operator: {unary_op.__name__}")
            operand_val = _safe_eval(operand)
            return _UNARY_OPS[unary_op](operand_val)  # type: ignore[no-any-return]
        case _:
            raise ValueError(f"Unsupported expression: {ast.dump(node)}")


# ---------------------------------------------------------------------------
# Public tool functions
# ---------------------------------------------------------------------------


def get_time() -> str:
    """Return the current date and time as an ISO-8601 formatted string.

    Example: ``"2026-06-03T14:22:31.123456+00:00"``.
    """
    return datetime.now(UTC).isoformat()


def echo(text: str) -> str:
    """Return *text* unchanged.

    An identity/echo tool useful for testing connectivity.
    """
    return text


def calculator(expression: str) -> str:
    """Evaluate a simple arithmetic expression safely and return the result.

    Supported operators: ``+``, ``-``, ``*``, ``/``, ``**``, and parentheses
    ``(`` ``)``.  Returns the numeric result on success (e.g. ``"42"`` or
    ``"3.14"``) or an error message prefixed with ``"Error: "`` on any failure.
    """
    try:
        tree = ast.parse(expression, mode="eval")
        result = _safe_eval(tree)
    except Exception as exc:
        return f"Error: {exc}"

    if isinstance(result, float) and result.is_integer():
        return str(int(result))
    return str(result)


def roll_dice(sides: int = 6) -> str:
    """Roll a single die with *sides* sides (default 6).

    Returns a string like ``"3"`` representing a random integer in
    ``[1, sides]``.  Raises ``ValueError`` if ``sides < 1``.
    """
    if sides < 1:
        raise ValueError(f"sides must be >= 1, got {sides}")
    return str(random.randint(1, sides))


# ---------------------------------------------------------------------------
# Convenience getter
# ---------------------------------------------------------------------------


def get_builtin_tools() -> list[Any]:
    """Return a fresh list of the four built-in tool callables.

    Each call returns a new ``list`` so callers can extend it without
    mutating shared state.  Compatible with
    ``build_agent(tools=get_builtin_tools())``.
    """
    return [get_time, echo, calculator, roll_dice]

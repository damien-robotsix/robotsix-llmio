"""Built-in example tools for robotsix-llmio agents.

Provides four simple tools — ``get_time``, ``echo``, ``calculator``,
``roll_dice`` — and a convenience getter ``get_builtin_tools`` that
returns them as a list ready for ``build_agent(tools=...)``.
"""

from __future__ import annotations

from ._builtins import get_builtin_tools

__all__ = ["get_builtin_tools"]

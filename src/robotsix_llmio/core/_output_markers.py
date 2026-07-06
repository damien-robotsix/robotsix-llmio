"""Shared pydantic-ai output-mode marker utilities.

Single source of truth for the ``_MARKERS`` tuple and the ``isinstance``
pattern used by both ``core/provider.py`` and ``claude_sdk/provider.py``.
"""

from __future__ import annotations

from typing import Any


def _is_output_type_marked(output_type: Any) -> bool:
    """Return True if *output_type* is a pydantic-ai output-mode marker
    (:class:`~pydantic_ai.PromptedOutput`, :class:`~pydantic_ai.ToolOutput`,
    :class:`~pydantic_ai.NativeOutput`) or a ``list``/``tuple`` that
    contains any such marker instance.
    """
    from pydantic_ai import NativeOutput, PromptedOutput, ToolOutput

    _MARKERS = (PromptedOutput, ToolOutput, NativeOutput)

    return isinstance(output_type, _MARKERS) or (
        isinstance(output_type, (list, tuple))
        and any(isinstance(entry, _MARKERS) for entry in output_type)
    )

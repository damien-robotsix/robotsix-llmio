"""Dependency-free HTML-to-plaintext helpers for downstream text consumers.

This module exposes a single stdlib-only utility, :func:`html_to_text`, that
collapses HTML markup down to whitespace-normalised plaintext suitable for LLM
prompts, email bodies, and other text-only sinks. It relies solely on the
standard library (:mod:`re` and :mod:`html`) so it can be shared by any
consumer of ``robotsix_llmio`` without pulling in an HTML-parsing dependency.
"""

from __future__ import annotations

import html
import re

_DROP_BLOCKS = re.compile(
    r"<(script|style|noscript|svg)\b[^>]*>.*?</\1>",
    re.IGNORECASE | re.DOTALL,
)
_TAG = re.compile(r"<[^>]+>")
_WHITESPACE = re.compile(r"\s+")


def html_to_text(html_text: str) -> str:
    """Strip HTML markup down to whitespace-collapsed plaintext.

    This is a dependency-free helper (stdlib :mod:`re` + :mod:`html` only) that
    performs, in order:

    1. Remove ``<script>``, ``<style>``, ``<noscript>``, and ``<svg>`` blocks
       *including their inner content* (case-insensitive, dot matches newline).
    2. Strip all remaining tags.
    3. ``html.unescape(...)`` the result to decode entities (e.g. ``&amp;``,
       ``&nbsp;``).
    4. Collapse all runs of whitespace to a single space and ``.strip()`` the
       ends.

    Args:
        html_text: The HTML markup to convert.

    Returns:
        The whitespace-collapsed plaintext rendering of ``html_text``.
    """
    text = _DROP_BLOCKS.sub(" ", html_text)
    text = _TAG.sub(" ", text)
    text = html.unescape(text)
    return _WHITESPACE.sub(" ", text).strip()

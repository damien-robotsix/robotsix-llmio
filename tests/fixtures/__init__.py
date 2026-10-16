"""Centralized, reusable test helpers for the robotsix-llmio suite.

This package is the single source of truth for mock *response* and mock
*object* construction that was previously inlined across the provider,
client, and cost test modules. It complements — it does not replace — the
per-area ``conftest.py`` files, which still own pytest *fixtures* (autouse
isolation, transport installers). See ``tests/FIXTURES.md`` for the full
map of fixtures and factories and when to reach for each.

Submodules:

* :mod:`tests.fixtures.responses` — ``httpx`` response / ``MockTransport``
  handler factories for the HTTP-backed layers (Langfuse cost, knowledge
  and refdocs clients).
* :mod:`tests.fixtures.builders` — ``SimpleNamespace`` / mock-object
  builders for the pydantic-ai response shapes read by the cost helpers
  (e.g. OpenRouter ``usage``).
"""

from __future__ import annotations

from tests.fixtures.builders import (
    make_openrouter_response,
    make_openrouter_usage,
)
from tests.fixtures.responses import (
    json_handler,
    make_error_response,
    make_knowledge_client_response,
    make_knowledge_result,
    make_langfuse_observation,
    make_langfuse_page,
    make_langfuse_trace,
    make_langfuse_traces_response,
    make_refdocs_response,
    make_refdocs_result,
    paged_handler,
)

__all__ = [
    "json_handler",
    "make_error_response",
    "make_knowledge_client_response",
    "make_knowledge_result",
    "make_langfuse_observation",
    "make_langfuse_page",
    "make_langfuse_trace",
    "make_langfuse_traces_response",
    "make_openrouter_response",
    "make_openrouter_usage",
    "make_refdocs_response",
    "make_refdocs_result",
    "paged_handler",
]

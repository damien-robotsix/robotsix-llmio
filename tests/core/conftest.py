"""Shared test helpers for core cost-log tests.

Pytest auto-discovers this conftest; the helper functions here are
imported explicitly by the test modules that need them.
"""

from __future__ import annotations

from datetime import UTC, datetime

import httpx

from robotsix_llmio.core import langfuse_cost as langfuse_cost_module
from robotsix_llmio.core.cost_log import CostWindow
from robotsix_llmio.core.langfuse_cost import LangfuseCostLogSource


def install_transport(monkeypatch, handler, module=langfuse_cost_module) -> list[httpx.Request]:
    """Patch ``httpx.Client`` so the adapter uses a ``MockTransport`` running
    *handler*. Returns a list that captures every request the adapter sends."""
    captured: list[httpx.Request] = []

    def _handler(request: httpx.Request) -> httpx.Response:
        captured.append(request)
        return handler(request)

    transport = httpx.MockTransport(_handler)
    real_client = httpx.Client

    def _client(*args, **kwargs):
        kwargs["transport"] = transport
        return real_client(*args, **kwargs)

    monkeypatch.setattr(module.httpx, "Client", _client)
    return captured


def make_window() -> CostWindow:
    return CostWindow(
        start=datetime(2026, 6, 3, 10, 0, tzinfo=UTC),
        end=datetime(2026, 6, 3, 11, 0, tzinfo=UTC),
    )


def make_adapter() -> LangfuseCostLogSource:
    return LangfuseCostLogSource(
        public_key="pub", secret_key="sec", base_url="https://lf.example.com"
    )

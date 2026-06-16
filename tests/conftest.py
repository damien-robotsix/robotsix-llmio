"""Top-level test helpers shared across all provider-cost test modules.

Pytest auto-discovers this conftest; the helper functions here are
imported explicitly by the test modules that need them.
"""

from __future__ import annotations

from datetime import UTC, datetime

import httpx

from robotsix_llmio.core.cost_log import CostWindow


def _window(start: str, end: str) -> CostWindow:
    return CostWindow(
        start=datetime.fromisoformat(start).replace(tzinfo=UTC),
        end=datetime.fromisoformat(end).replace(tzinfo=UTC),
    )


def _mock_client_factory(monkeypatch, module, handler):
    """Patch *module*.httpx.Client to use a MockTransport(handler).

    ``module.httpx`` is the shared httpx module, so patching its ``Client``
    affects every reference — capture the real class FIRST so the factory
    doesn't recurse into itself.
    """
    real_client = httpx.Client

    def _make(*args, **kwargs):
        kwargs.pop("transport", None)
        return real_client(transport=httpx.MockTransport(handler), **kwargs)

    monkeypatch.setattr(module.httpx, "Client", _make)

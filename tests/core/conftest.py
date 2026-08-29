"""Shared test helpers for core tests.

Pytest auto-discovers this conftest; the helper functions here are
imported explicitly by the test modules that need them.
"""

from __future__ import annotations

import base64
import os
from datetime import UTC, datetime

import httpx
import pytest

from robotsix_llmio.config.tier import TierConfig, TierLevelConfig
from robotsix_llmio.core import langfuse_cost as langfuse_cost_module
from robotsix_llmio.core.cost_log import CostWindow
from robotsix_llmio.core.langfuse_cost import LangfuseCostLogSource

# --------------------------------------------------------------------------- #
#  Tier-fallback test helpers                                                 #
# --------------------------------------------------------------------------- #

_L1_TF_CFG = TierLevelConfig(model="claudeSDK-opus")
_L2_TF_CFG = TierLevelConfig(model="claudeSDK-haiku")
_L3_TF_CFG = TierLevelConfig(model="claudeSDK-sonnet")
_L4_TF_CFG = TierLevelConfig(model="claudeSDK-claude-fable-5")
_L5_TF_CFG = TierLevelConfig(model="claudeSDK-frontier")

STD_TIER_CONFIG = TierConfig(
    level1=_L1_TF_CFG,
    level2=_L2_TF_CFG,
    level3=_L3_TF_CFG,
    level4=_L4_TF_CFG,
    level5=_L5_TF_CFG,
)


def tf_factory_that_succeeds(
    result: str = "ok",
    *,
    tracking: dict | None = None,
    expected_tier: str | None = None,
):
    """Return a factory that returns a callable that succeeds."""

    def factory(tlc: TierLevelConfig):
        if tracking is not None:
            tracking.setdefault("factory_calls", []).append(tlc.model_name)
        if expected_tier is not None:
            assert tlc.model_name == expected_tier, (
                f"expected {expected_tier}, got {tlc.model_name}"
            )

        def fn():
            return result

        return fn

    return factory


def tf_failing_factory(
    exception: type[Exception] | Exception,
    *,
    fail_count: int = 1,
    tracking: dict | None = None,
):
    """Return a factory whose callable raises *exception* the first *fail_count*
    times it is invoked (global counter across all factory calls)."""
    counter = {"remaining": fail_count}

    def factory(tlc: TierLevelConfig):
        if tracking is not None:
            tracking.setdefault("factory_calls", []).append(tlc.model_name)

        def fn():
            if counter["remaining"] > 0:
                counter["remaining"] -= 1
                if isinstance(exception, type):
                    raise exception("boom")
                raise exception
            return "finally-ok"

        return fn

    return factory


def tf_exhausted_failing_factory(
    exception: type[Exception] | Exception = RuntimeError,
    *,
    tracking: dict | None = None,
):
    """Return a factory whose callable always raises."""

    def factory(tlc: TierLevelConfig):
        if tracking is not None:
            tracking.setdefault("factory_calls", []).append(tlc.model_name)

        def fn():
            if isinstance(exception, type):
                raise exception("boom")
            raise exception

        return fn

    return factory


def install_transport(
    monkeypatch, handler, module=langfuse_cost_module
) -> list[httpx.Request]:
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


def install_timeout_transport(monkeypatch, handler, module) -> None:
    """Patch ``module.timeout_http_client`` so the client under test uses a
    ``MockTransport`` running *handler*."""
    transport = httpx.MockTransport(handler)

    def _fake_timeout_client() -> httpx.AsyncClient:
        return httpx.AsyncClient(transport=transport)

    monkeypatch.setattr(module, "timeout_http_client", _fake_timeout_client)


def install_async_transport(monkeypatch, handler, module) -> list[httpx.Request]:
    """Patch ``httpx.AsyncClient`` so the adapter uses a ``MockTransport``
    running *handler*. Returns a list that captures every request the
    adapter sends."""
    captured: list[httpx.Request] = []

    def _handler(request: httpx.Request) -> httpx.Response:
        captured.append(request)
        return handler(request)

    transport = httpx.MockTransport(_handler)
    real_async_client = httpx.AsyncClient

    def _client(*args, **kwargs):
        kwargs["transport"] = transport
        return real_async_client(*args, **kwargs)

    monkeypatch.setattr(module.httpx, "AsyncClient", _client)
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


# --------------------------------------------------------------------------- #
#  Retry test helpers                                                         #
# --------------------------------------------------------------------------- #


class _HTTPErr(Exception):
    def __init__(self, status_code):
        self.status_code = status_code


# --------------------------------------------------------------------------- #
#  Tier-fallback sleep helpers (avoids real waits in tests)                   #
# --------------------------------------------------------------------------- #


def _noop_sleep(_d: float) -> None:
    return None


async def _anoop_sleep(_d: float) -> None:
    return None


# --------------------------------------------------------------------------- #
#  Live-test Langfuse helpers (shared across OpenRouter / Claude SDK tests)   #
# --------------------------------------------------------------------------- #


def _langfuse_creds() -> tuple[str | None, str | None, str]:
    return (
        os.environ.get("LANGFUSE_PUBLIC_KEY"),
        os.environ.get("LANGFUSE_SECRET_KEY"),
        os.environ.get("LANGFUSE_BASE_URL", "https://cloud.langfuse.com"),
    )


def _require() -> None:
    pk, sk, _ = _langfuse_creds()
    if not (pk and sk):
        pytest.skip("LANGFUSE_PUBLIC_KEY / LANGFUSE_SECRET_KEY not set")
    if not os.environ.get("OPENROUTER_API_KEY"):
        pytest.skip("OPENROUTER_API_KEY not set")


def _langfuse_traces(session_id: str) -> list[dict] | None:
    """GET the Langfuse traces for *session_id*, or None on a failed request."""
    pk, sk, base = _langfuse_creds()
    auth = base64.b64encode(f"{pk}:{sk}".encode()).decode()
    with httpx.Client(timeout=20) as client:
        resp = client.get(
            f"{base.rstrip('/')}/api/public/traces",
            params={"sessionId": session_id, "limit": 10},
            headers={"Authorization": f"Basic {auth}"},
        )
    if resp.status_code != 200:
        return None
    return resp.json().get("data", [])


def _langfuse_get(path: str, params: dict) -> dict | None:
    """Authenticated GET to the Langfuse public API; None on failure."""
    pk, sk, base = _langfuse_creds()
    auth = base64.b64encode(f"{pk}:{sk}".encode()).decode()
    with httpx.Client(timeout=20) as client:
        resp = client.get(
            f"{base.rstrip('/')}{path}",
            params=params,
            headers={"Authorization": f"Basic {auth}"},
        )
    return resp.json() if resp.status_code == 200 else None

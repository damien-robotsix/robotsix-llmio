"""Live Langfuse multi-tenant test: cross-project trace isolation.

On-demand only (``live`` marker). Requires two sets of Langfuse credentials
(``LANGFUSE_PUBLIC_KEY`` / ``LANGFUSE_SECRET_KEY`` for project A,
``LANGFUSE_PUBLIC_KEY_2`` / ``LANGFUSE_SECRET_KEY_2`` for project B) plus
``OPENROUTER_API_KEY``. Skips with a message when the second credential set
is absent, so the single-tenant CI path stays green.
"""

from __future__ import annotations

import base64
import os
import time
import uuid

import pytest

from robotsix_llmio.config.tier import (
    FALLBACK_LEVEL1,
    FALLBACK_LEVEL2,
    FALLBACK_LEVEL3,
    ProviderSlotConfig,
    TierConfig,
)

# OpenRouter bindings in the DEFAULT slot: these live tests exercise the
# OpenRouter transport directly, with no failover machinery involved.
_OPENROUTER_TIERS = TierConfig(
    default=ProviderSlotConfig(
        level1=FALLBACK_LEVEL1,
        level2=FALLBACK_LEVEL2,
        level3=FALLBACK_LEVEL3,
    ),
)


def _langfuse_traces_for_project(
    session_id: str, pk: str, sk: str, base_url: str
) -> list[dict] | None:
    """GET traces for *session_id* from a specific Langfuse project."""
    import httpx

    auth = base64.b64encode(f"{pk}:{sk}".encode()).decode()
    with httpx.Client(timeout=20) as client:
        resp = client.get(
            f"{base_url.rstrip('/')}/api/public/traces",
            params={"sessionId": session_id, "limit": 10},
            headers={"Authorization": f"Basic {auth}"},
        )
    if resp.status_code != 200:
        return None
    return resp.json().get("data", [])


@pytest.mark.live
@pytest.mark.timeout(120)
def test_multi_tenant_no_cross_project_leakage() -> None:
    """Two Langfuse projects, each with its own session — verify each session's
    traces appear *only* in its owning project and never in the other.

    Requires ``LANGFUSE_PUBLIC_KEY`` / ``LANGFUSE_SECRET_KEY`` for the default
    project AND ``LANGFUSE_PUBLIC_KEY_2`` / ``LANGFUSE_SECRET_KEY_2`` for the
    second project.  Skips with a message when the second set is absent, so the
    single-tenant CI path stays green.
    """
    pk_a = os.environ.get("LANGFUSE_PUBLIC_KEY")
    sk_a = os.environ.get("LANGFUSE_SECRET_KEY")
    pk_b = os.environ.get("LANGFUSE_PUBLIC_KEY_2")
    sk_b = os.environ.get("LANGFUSE_SECRET_KEY_2")

    if not (pk_a and sk_a):
        pytest.skip("LANGFUSE_PUBLIC_KEY / LANGFUSE_SECRET_KEY not set")
    if not (pk_b and sk_b):
        pytest.skip(
            "Multi-tenant live test requires LANGFUSE_PUBLIC_KEY_2 and "
            "LANGFUSE_SECRET_KEY_2"
        )
    if not os.environ.get("OPENROUTER_API_KEY"):
        pytest.skip("OPENROUTER_API_KEY not set")

    base = os.environ.get("LANGFUSE_BASE_URL", "https://cloud.langfuse.com")

    # Reset tracing state to known-clean (earlier live tests may have
    # registered projects, etc.).
    import robotsix_llmio.core.tracing as _t
    from robotsix_llmio.core import (
        flush_tracing,
        langfuse_project,
        langfuse_session,
        setup_langfuse_tracing,
    )
    from robotsix_llmio.openrouter._deepseek_provider import OpenRouterDeepseekProvider

    _t._provider = None
    _t._projects.clear()
    _t._default_public_key = None
    _t._trace_routing.clear()

    assert (
        setup_langfuse_tracing(public_key=pk_a, secret_key=sk_a, base_url=base) is True
    )
    assert (
        setup_langfuse_tracing(public_key=pk_b, secret_key=sk_b, base_url=base) is True
    )

    session_a = f"llmio-mt-a-{uuid.uuid4().hex[:12]}"
    session_b = f"llmio-mt-b-{uuid.uuid4().hex[:12]}"

    provider = OpenRouterDeepseekProvider()

    # Session A → project A
    agent_a = provider.build_agent(
        level=1,
        tier_config=_OPENROUTER_TIERS,
        system_prompt="You are concise. Answer with just the number.",
        name="mt-agent-a",
    )
    try:
        with langfuse_project(pk_a), langfuse_session(session_a):
            result = provider.call_with_retry(
                lambda: agent_a.run_sync(
                    "What is 3+3?", model_settings={"max_tokens": 20}
                )
            )
        assert "6" in str(result.output)
    finally:
        agent_a.close()

    # Session B → project B
    agent_b = provider.build_agent(
        level=1,
        tier_config=_OPENROUTER_TIERS,
        system_prompt="You are concise. Answer with just the number.",
        name="mt-agent-b",
    )
    try:
        with langfuse_project(pk_b), langfuse_session(session_b):
            result = provider.call_with_retry(
                lambda: agent_b.run_sync(
                    "What is 4+4?", model_settings={"max_tokens": 20}
                )
            )
        assert "8" in str(result.output)
    finally:
        agent_b.close()

    flush_tracing()

    # Poll each project for its own session's traces.
    traces_a: list[dict] | None = None
    traces_b: list[dict] | None = None
    for _ in range(15):
        if traces_a is None:
            traces_a = _langfuse_traces_for_project(session_a, pk_a, sk_a, base)
        if traces_b is None:
            traces_b = _langfuse_traces_for_project(session_b, pk_b, sk_b, base)
        if traces_a and traces_b:
            break
        time.sleep(4)

    assert traces_a, f"no traces in project A for session {session_a}"
    assert traces_b, f"no traces in project B for session {session_b}"

    # Verify NO cross-project leakage.
    traces_a_in_b = _langfuse_traces_for_project(session_a, pk_b, sk_b, base)
    traces_b_in_a = _langfuse_traces_for_project(session_b, pk_a, sk_a, base)

    assert not traces_a_in_b, f"session A traces leaked into project B: {traces_a_in_b}"
    assert not traces_b_in_a, f"session B traces leaked into project A: {traces_b_in_a}"

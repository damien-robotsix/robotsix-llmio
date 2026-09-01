"""Live Langfuse round-trip: OpenRouter DeepSeek provider tests.

On-demand only (``live`` marker). Skips unless both ``LANGFUSE_PUBLIC_KEY`` /
``LANGFUSE_SECRET_KEY`` and ``OPENROUTER_API_KEY`` are set. Uses the cheap tier
and a tiny ``max_tokens`` to keep the spend negligible.
"""

from __future__ import annotations

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
from tests.core.conftest import (
    _langfuse_creds,
    _langfuse_get,
    _langfuse_traces,
    _require,
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


@pytest.mark.live
@pytest.mark.timeout(120)
def test_langfuse_trace_roundtrip_has_cost() -> None:
    """A real provider call, grouped under a unique session, produces a Langfuse
    trace whose ``totalCost`` is populated from the per-call cost the model
    stamps on the span."""
    _require()

    from robotsix_llmio.core import (
        flush_tracing,
        langfuse_session,
        setup_langfuse_tracing,
    )
    from robotsix_llmio.openrouter._deepseek_provider import OpenRouterDeepseekProvider

    assert setup_langfuse_tracing() is True, "tracing should configure with creds"

    session_id = f"llmio-livetest-{uuid.uuid4().hex[:12]}"
    provider = OpenRouterDeepseekProvider()
    agent = provider.build_agent(
        level=1,
        tier_config=_OPENROUTER_TIERS,
        system_prompt="You are concise. Answer with just the number.",
        name="tracing-livetest",
    )
    try:
        with langfuse_session(session_id):
            result = provider.call_with_retry(
                lambda: agent.run_sync(
                    "What is 2+2?", model_settings={"max_tokens": 20}
                )
            )
        assert "4" in str(result.output)
    finally:
        agent.close()

    flush_tracing()

    # Langfuse ingestion is asynchronous — poll for the trace to appear.
    traces: list[dict] | None = None
    for _ in range(15):
        traces = _langfuse_traces(session_id)
        if traces:
            break
        time.sleep(4)

    assert traces, f"no Langfuse trace for session {session_id!r} after polling"
    total_cost = sum(float(t.get("totalCost") or 0) for t in traces)
    assert total_cost > 0, (
        f"trace landed but totalCost={total_cost} (expected > 0 — cost should "
        f"flow from the model span's langfuse.observation.cost_details)"
    )


@pytest.mark.live
@pytest.mark.timeout(120)
def test_langfuse_trace_tool_and_subagent() -> None:
    """A run that uses a tool AND delegates to a subagent, so we can see how
    nested tool/agent spans display in Langfuse.

    The outer agent has a plain ``add`` tool plus an async ``consult_expert``
    tool that runs a second (sub)agent. With ``instrument_all()`` the inner run
    nests under the outer tool span, and every model call is a generation — so
    the Langfuse trace shows: coordinator run → add tool → consult_expert tool →
    subagent run → its generation. Prints the trace URL for inspection.
    """
    _require()  # LANGFUSE_* + OPENROUTER_API_KEY

    from robotsix_llmio.core import (
        flush_tracing,
        langfuse_session,
        setup_langfuse_tracing,
    )
    from robotsix_llmio.openrouter._deepseek_provider import OpenRouterDeepseekProvider

    assert setup_langfuse_tracing() is True

    provider = OpenRouterDeepseekProvider()
    subagent = provider.build_agent(
        level=1,
        tier_config=_OPENROUTER_TIERS,
        system_prompt="You are a physics expert. Answer in one short sentence.",
        name="subagent-physics",
    )

    def add(a: int, b: int) -> int:
        """Add two integers."""
        return a + b

    async def consult_expert(question: str) -> str:
        """Delegate a question to a specialist subagent and return its answer."""
        run = await subagent.run(question)
        return str(run.output)

    outer = provider.build_agent(
        level=1,
        tier_config=_OPENROUTER_TIERS,
        system_prompt=(
            "You coordinate. Use the add tool for arithmetic and the "
            "consult_expert tool for science questions."
        ),
        tools=[add, consult_expert],
        name="coordinator",
    )

    session_id = f"llmio-livetest-subagent-{uuid.uuid4().hex[:12]}"
    try:
        with langfuse_session(session_id):
            result = provider.call_with_retry(
                lambda: outer.run_sync(
                    "First use add to compute 21 + 21. Then use consult_expert "
                    "to ask why the sky is blue. Report both answers."
                )
            )
        assert "42" in str(result.output)
    finally:
        outer.close()
        subagent.close()

    flush_tracing()

    traces: list[dict] | None = None
    for _ in range(15):
        traces = _langfuse_traces(session_id)
        if traces:
            break
        time.sleep(4)
    assert traces, f"no Langfuse trace for session {session_id!r} after polling"

    trace = traces[0]
    trace_id = trace.get("id")
    obs_body = _langfuse_get(
        "/api/public/observations", {"traceId": trace_id, "limit": 100}
    )
    observations = (obs_body or {}).get("data", [])
    by_type: dict[str, int] = {}
    for o in observations:
        by_type[o.get("type", "?")] = by_type.get(o.get("type", "?"), 0) + 1

    # Surface the structure for inspection.
    _, _, base = _langfuse_creds()
    project_id = trace.get("projectId")
    url = (
        f"{base.rstrip('/')}/project/{project_id}/traces/{trace_id}"
        if project_id
        else f"{base.rstrip('/')} (search sessionId={session_id})"
    )
    print(f"\n[langfuse] session={session_id}")
    print(f"[langfuse] trace={trace_id} totalCost={trace.get('totalCost')}")
    print(f"[langfuse] observations by type: {by_type}")
    print(f"[langfuse] view: {url}")
    for o in sorted(observations, key=lambda x: x.get("startTime") or ""):
        print(f"    - {o.get('type'):10} {o.get('name')}")

    # Tool + subagent => several observations incl. >1 generation (outer + inner).
    assert len(observations) >= 3, f"expected a rich trace, got {by_type}"
    generations = by_type.get("GENERATION", 0)
    assert generations >= 2, (
        f"expected >=2 generations (outer + subagent), got {by_type}"
    )


@pytest.mark.live
@pytest.mark.timeout(120)
def test_langfuse_trace_url_resolves_to_real_trace() -> None:
    """langfuse_trace_url builds a UI URL with the project's real id, pointing at
    a trace that actually exists. The project id is discovered from the API, so
    no LANGFUSE_PROJECT_ID env is required."""
    _require()  # LANGFUSE_* + OPENROUTER_API_KEY

    from robotsix_llmio.core import (
        flush_tracing,
        langfuse_trace_url,
        make_session_id,
        setup_langfuse_tracing,
        start_trace,
    )
    from robotsix_llmio.openrouter._deepseek_provider import OpenRouterDeepseekProvider

    projects = _langfuse_get("/api/public/projects", {})
    assert projects and projects.get("data"), "could not discover the Langfuse project"
    project_id = projects["data"][0]["id"]

    # Register (or backfill) the project id so langfuse_trace_url can use it.
    assert setup_langfuse_tracing(project_id=project_id) is True
    _, _, base = _langfuse_creds()

    provider = OpenRouterDeepseekProvider()
    agent = provider.build_agent(
        level=1,
        tier_config=_OPENROUTER_TIERS,
        system_prompt="Concise.",
        name="url-test",
    )
    trace_id: str | None = None
    try:
        with start_trace("url-trace", session_id=make_session_id("urltest")) as root:
            provider.call_with_retry(
                lambda: agent.run_sync(
                    "2+2? Just the number.", model_settings={"max_tokens": 20}
                )
            )
            trace_id = root.trace_id
    finally:
        agent.close()
    flush_tracing()

    assert trace_id, "start_trace should expose a trace_id"
    url = langfuse_trace_url(trace_id)
    assert url == f"{base.rstrip('/')}/project/{project_id}/traces/{trace_id}", url

    # The trace the URL points at must actually exist (poll for ingestion).
    found = None
    for _ in range(12):
        found = _langfuse_get(f"/api/public/traces/{trace_id}", {})
        if found:
            break
        time.sleep(4)
    assert found and found.get("id") == trace_id, "URL points at a missing trace"


@pytest.mark.live
@pytest.mark.timeout(120)
def test_langfuse_cost_log_source_reads_back_logged_cost() -> None:
    """A freshly logged session's cost is readable back through the neutral
    ``CostLogSource`` port (the read-side counterpart to the OTLP write seam)."""
    _require()  # LANGFUSE_* + OPENROUTER_API_KEY

    import datetime as _dt

    from robotsix_llmio.core import (
        CostWindow,
        LangfuseCostLogSource,
        flush_tracing,
        langfuse_session,
        setup_langfuse_tracing,
    )
    from robotsix_llmio.openrouter._deepseek_provider import OpenRouterDeepseekProvider

    assert setup_langfuse_tracing() is True, "tracing should configure with creds"

    start = _dt.datetime.now(_dt.UTC) - _dt.timedelta(minutes=5)
    session_id = f"llmio-livetest-costlog-{uuid.uuid4().hex[:12]}"
    provider = OpenRouterDeepseekProvider()
    agent = provider.build_agent(
        level=1,
        tier_config=_OPENROUTER_TIERS,
        system_prompt="You are concise. Answer with just the number.",
        name="costlog-livetest",
    )
    try:
        with langfuse_session(session_id):
            result = provider.call_with_retry(
                lambda: agent.run_sync(
                    "What is 2+2?", model_settings={"max_tokens": 20}
                )
            )
        assert "4" in str(result.output)
    finally:
        agent.close()

    flush_tracing()

    # Langfuse ingestion is asynchronous — poll for the trace to appear.
    traces: list[dict] | None = None
    for _ in range(15):
        traces = _langfuse_traces(session_id)
        if traces:
            break
        time.sleep(4)
    assert traces, f"no Langfuse trace for session {session_id!r} after polling"

    pk, sk, base = _langfuse_creds()
    source = LangfuseCostLogSource(public_key=pk, secret_key=sk, base_url=base)
    end = _dt.datetime.now(_dt.UTC) + _dt.timedelta(minutes=1)
    logged = source.fetch_logged_cost(CostWindow(start=start, end=end))
    assert logged.total_cost > 0, (
        f"expected total_cost > 0 read back via LangfuseCostLogSource, "
        f"got {logged.total_cost}"
    )

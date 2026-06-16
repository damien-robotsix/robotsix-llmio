"""Provider-agnostic LLM I/O base.

All submodule imports are deferred via PEP 562 ``__getattr__`` so that
importing lightweight helpers (e.g. ``core.retry``) does not eagerly pull in
pydantic-ai or OpenTelemetry at module load time.
"""

from __future__ import annotations

from typing import Any

__all__ = [
    "DEFAULT_TOLERANCE",
    "AgentHandle",
    "CostLogSource",
    "CostRecord",
    "CostWindow",
    "Discrepancy",
    "LLMProvider",
    "LangfuseCostLogSource",
    "LangfuseReadClient",
    "LoggedCost",
    "ProviderCost",
    "ProviderCostSource",
    "Tier",
    "TraceSpan",
    "acall_with_retry",
    "acall_with_retry_and_fallback",
    "active_routing_key",
    "arun_agent",
    "build_agent",
    "call_with_retry",
    "call_with_retry_and_fallback",
    "current_session",
    "flush_tracing",
    "get_provider",
    "get_recording_span",
    "get_tracer",
    "html_to_text",
    "install_signal_handlers",
    "is_rate_limited",
    "is_transient",
    "langfuse_project",
    "langfuse_session",
    "langfuse_trace_url",
    "make_session_id",
    "reconcile",
    "register_provider",
    "run_agent",
    "setup_langfuse_tracing",
    "start_span",
    "start_trace",
    "timeout_http_client",
]


def __getattr__(name: str) -> Any:  # PEP 562 — lazy heavy imports
    if name in ("AgentHandle", "build_agent"):
        from . import agent

        return getattr(agent, name)
    if name in ("CostLogSource", "CostRecord", "CostWindow", "LoggedCost"):
        from . import cost_log

        return getattr(cost_log, name)
    if name in ("get_provider", "register_provider"):
        from . import factory

        return getattr(factory, name)
    if name == "timeout_http_client":
        from . import http

        return getattr(http, name)
    if name == "LangfuseReadClient":
        from . import langfuse_client

        return getattr(langfuse_client, name)
    if name == "LangfuseCostLogSource":
        from . import langfuse_cost

        return getattr(langfuse_cost, name)
    if name in ("LLMProvider", "Tier"):
        from . import provider

        return getattr(provider, name)
    if name in (
        "DEFAULT_TOLERANCE",
        "Discrepancy",
        "ProviderCost",
        "ProviderCostSource",
        "reconcile",
    ):
        from . import provider_cost

        return getattr(provider_cost, name)
    if name in (
        "acall_with_retry",
        "acall_with_retry_and_fallback",
        "call_with_retry",
        "call_with_retry_and_fallback",
        "is_rate_limited",
        "is_transient",
    ):
        from . import retry

        return getattr(retry, name)
    if name in ("arun_agent", "run_agent"):
        from . import run

        return getattr(run, name)
    if name == "html_to_text":
        from . import text_utils

        return getattr(text_utils, name)
    if name in (
        "TraceSpan",
        "active_routing_key",
        "current_session",
        "flush_tracing",
        "get_recording_span",
        "get_tracer",
        "install_signal_handlers",
        "langfuse_project",
        "langfuse_session",
        "langfuse_trace_url",
        "make_session_id",
        "setup_langfuse_tracing",
        "start_span",
        "start_trace",
    ):
        from . import tracing

        return getattr(tracing, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

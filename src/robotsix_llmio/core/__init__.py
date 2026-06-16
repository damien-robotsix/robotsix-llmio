"""Provider-agnostic LLM I/O base.

All submodule imports are deferred via PEP 562 ``__getattr__`` so that
importing lightweight helpers (e.g. ``core.retry``) does not eagerly pull in
pydantic-ai or OpenTelemetry at module load time.
"""

from __future__ import annotations

from typing import Any

__all__ = [
    "DEFAULT_TOLERANCE",
    "LEGACY_TIER_MAP",
    "LEVEL1_DEFAULT",
    "LEVEL2_DEFAULT",
    "LEVEL3_DEFAULT",
    "MODEL_LEVEL_TO_TIER",
    "PROVIDER_MODELS",
    "TRANSPORT_ALIASES",
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
    "TierConfig",
    "TierConfigLoadError",
    "TierLevel",
    "TierLevelConfig",
    "TraceSpan",
    "UnknownModelError",
    "acall_with_retry",
    "acall_with_retry_and_fallback",
    "acall_with_tier_fallback",
    "active_routing_key",
    "arun_agent",
    "build_agent",
    "call_with_retry",
    "call_with_retry_and_fallback",
    "call_with_tier_fallback",
    "create_model",
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
    "load_tier_config",
    "make_session_id",
    "reconcile",
    "register_provider",
    "run_agent",
    "setup_langfuse_tracing",
    "start_span",
    "start_trace",
    "timeout_http_client",
    "validate_model",
]


def __getattr__(name: str) -> Any:  # PEP 562 — lazy heavy imports
    if name == "AgentHandle":
        from . import agent

        return agent.AgentHandle
    if name == "build_agent":
        from . import agent

        return agent.build_agent
    if name == "CostLogSource":
        from . import cost_log

        return cost_log.CostLogSource
    if name == "CostRecord":
        from . import cost_log

        return cost_log.CostRecord
    if name == "CostWindow":
        from . import cost_log

        return cost_log.CostWindow
    if name == "LoggedCost":
        from . import cost_log

        return cost_log.LoggedCost
    if name == "get_provider":
        from . import factory

        return factory.get_provider
    if name == "register_provider":
        from . import factory

        return factory.register_provider
    if name == "timeout_http_client":
        from . import http

        return http.timeout_http_client
    if name == "LangfuseReadClient":
        from . import langfuse_client

        return langfuse_client.LangfuseReadClient
    if name == "LangfuseCostLogSource":
        from . import langfuse_cost

        return langfuse_cost.LangfuseCostLogSource
    if name == "LLMProvider":
        from . import provider

        return provider.LLMProvider
    if name == "Tier":
        from . import provider

        return provider.Tier
    if name == "LEVEL1_DEFAULT":
        from robotsix_llmio.config import tier as _config_tier

        return _config_tier.LEVEL1_DEFAULT
    if name == "LEVEL2_DEFAULT":
        from robotsix_llmio.config import tier as _config_tier

        return _config_tier.LEVEL2_DEFAULT
    if name == "LEVEL3_DEFAULT":
        from robotsix_llmio.config import tier as _config_tier

        return _config_tier.LEVEL3_DEFAULT
    if name == "LEGACY_TIER_MAP":
        import warnings

        warnings.warn(
            "LEGACY_TIER_MAP is deprecated. Use TierConfig.for_level() instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        from robotsix_llmio.config import tier as _config_tier

        return _config_tier.LEGACY_TIER_MAP
    if name == "TierConfig":
        from robotsix_llmio.config import tier as _config_tier

        return _config_tier.TierConfig
    if name == "TierLevel":
        from robotsix_llmio.config import tier as _config_tier

        return _config_tier.TierLevel
    if name == "TierLevelConfig":
        from robotsix_llmio.config import tier as _config_tier

        return _config_tier.TierLevelConfig
    if name == "TierConfigLoadError":
        from robotsix_llmio.config import loader as _config_loader

        return _config_loader.TierConfigLoadError
    if name == "load_tier_config":
        from robotsix_llmio.config import loader as _config_loader

        return _config_loader.load_tier_config
    if name == "MODEL_LEVEL_TO_TIER":
        from robotsix_llmio.config import transport as _config_transport

        return _config_transport.MODEL_LEVEL_TO_TIER
    if name == "PROVIDER_MODELS":
        from robotsix_llmio.config import model_registry as _config_model_registry

        return _config_model_registry.PROVIDER_MODELS
    if name == "TRANSPORT_ALIASES":
        from robotsix_llmio.config import transport as _config_transport

        return _config_transport.TRANSPORT_ALIASES
    if name == "create_model":
        from robotsix_llmio.config import factory as _config_factory

        return _config_factory.create_model
    if name == "UnknownModelError":
        from robotsix_llmio.config import model_registry as _config_model_registry

        return _config_model_registry.UnknownModelError
    if name == "validate_model":
        from robotsix_llmio.config import model_registry as _config_model_registry

        return _config_model_registry.validate_model
    if name == "DEFAULT_TOLERANCE":
        from . import provider_cost

        return provider_cost.DEFAULT_TOLERANCE
    if name == "Discrepancy":
        from . import provider_cost

        return provider_cost.Discrepancy
    if name == "ProviderCost":
        from . import provider_cost

        return provider_cost.ProviderCost
    if name == "ProviderCostSource":
        from . import provider_cost

        return provider_cost.ProviderCostSource
    if name == "reconcile":
        from . import provider_cost

        return provider_cost.reconcile
    if name == "acall_with_retry":
        from . import retry

        return retry.acall_with_retry
    if name == "acall_with_retry_and_fallback":
        from . import retry

        return retry.acall_with_retry_and_fallback
    if name == "call_with_retry":
        from . import retry

        return retry.call_with_retry
    if name == "call_with_retry_and_fallback":
        from . import retry

        return retry.call_with_retry_and_fallback
    if name == "acall_with_tier_fallback":
        from . import tier_fallback

        return tier_fallback.acall_with_tier_fallback
    if name == "call_with_tier_fallback":
        from . import tier_fallback

        return tier_fallback.call_with_tier_fallback
    if name == "is_rate_limited":
        from . import retry

        return retry.is_rate_limited
    if name == "is_transient":
        from . import retry

        return retry.is_transient
    if name == "arun_agent":
        from . import run

        return run.arun_agent
    if name == "run_agent":
        from . import run

        return run.run_agent
    if name == "html_to_text":
        from . import text_utils

        return text_utils.html_to_text
    if name == "TraceSpan":
        from . import tracing

        return tracing.TraceSpan
    if name == "active_routing_key":
        from . import tracing

        return tracing.active_routing_key
    if name == "current_session":
        from . import tracing

        return tracing.current_session
    if name == "flush_tracing":
        from . import tracing

        return tracing.flush_tracing
    if name == "get_recording_span":
        from . import tracing

        return tracing.get_recording_span
    if name == "get_tracer":
        from . import tracing

        return tracing.get_tracer
    if name == "install_signal_handlers":
        from . import tracing

        return tracing.install_signal_handlers
    if name == "langfuse_project":
        from . import tracing

        return tracing.langfuse_project
    if name == "langfuse_session":
        from . import tracing

        return tracing.langfuse_session
    if name == "langfuse_trace_url":
        from . import tracing

        return tracing.langfuse_trace_url
    if name == "make_session_id":
        from . import tracing

        return tracing.make_session_id
    if name == "setup_langfuse_tracing":
        from . import tracing

        return tracing.setup_langfuse_tracing
    if name == "start_span":
        from . import tracing

        return tracing.start_span
    if name == "start_trace":
        from . import tracing

        return tracing.start_trace
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

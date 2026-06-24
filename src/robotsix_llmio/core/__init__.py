"""Provider-agnostic LLM I/O base.

All submodule imports are deferred via PEP 562 ``__getattr__`` so that
importing lightweight helpers (e.g. ``core.retry``) does not eagerly pull in
pydantic-ai or OpenTelemetry at module load time.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

# Static re-declaration of every lazily-exported name (see ``__getattr__``
# below). These imports run ONLY under static analysis (``TYPE_CHECKING`` is
# False at runtime), so they add no import-time cost and preserve the PEP 562
# lazy-loading behaviour — but they let type checkers, IDEs, and CodeQL see
# each ``__all__`` entry as a defined module global. Without this, CodeQL's
# ``py/undefined-export`` query flags every ``__all__`` name as "exported but
# not defined" (it cannot model PEP 562 dynamic exports), failing the
# code-scanning check on any PR that adds a new export. Keep this block in
# sync with ``__all__`` and ``__getattr__``.
if TYPE_CHECKING:
    from robotsix_llmio.config.factory import create_model
    from robotsix_llmio.config.loader import TierConfigLoadError, load_tier_config
    from robotsix_llmio.config.tier import (
        LEVEL1_DEFAULT,
        LEVEL2_DEFAULT,
        LEVEL3_DEFAULT,
        TierConfig,
        TierLevel,
        TierLevelConfig,
    )

    from .agent import AgentHandle, build_agent
    from .cost_log import CostLogSource, CostRecord, CostWindow, LoggedCost
    from .factory import (
        build_agent_for_level,
        default_tier_config,
        get_provider_for_identifier,
        get_provider_for_level,
    )
    from .http import timeout_http_client
    from .identifier import (
        MalformedIdentifierError,
        ParsedIdentifier,
        parse_model_identifier,
    )
    from .langfuse_async_client import AsyncLangfuseReadClient
    from .langfuse_client import LangfuseReadClient
    from .langfuse_cost import LangfuseCostLogSource
    from .provider import LLMProvider
    from .provider_cost import (
        DEFAULT_TOLERANCE,
        Discrepancy,
        ProviderCost,
        ProviderCostSource,
        reconcile,
    )
    from .retry import (
        acall_with_retry,
        acall_with_retry_and_fallback,
        call_with_retry,
        call_with_retry_and_fallback,
        is_rate_limited,
        is_transient,
    )
    from .run import arun_agent, run_agent
    from .sqlite_utils import add_column_if_missing, run_additive_migrations
    from .text_utils import html_to_text
    from .tier_fallback import acall_with_tier_fallback, call_with_tier_fallback
    from .tracing import (
        TraceSpan,
        active_routing_key,
        current_session,
        flush_tracing,
        get_recording_span,
        get_tracer,
        install_signal_handlers,
        langfuse_project,
        langfuse_session,
        langfuse_trace_url,
        make_session_id,
        setup_langfuse_tracing,
        start_span,
        start_trace,
    )

__all__ = [
    "DEFAULT_TOLERANCE",
    "LEVEL1_DEFAULT",
    "LEVEL2_DEFAULT",
    "LEVEL3_DEFAULT",
    "AgentHandle",
    "AsyncLangfuseReadClient",
    "CostLogSource",
    "CostRecord",
    "CostWindow",
    "Discrepancy",
    "LLMProvider",
    "LangfuseCostLogSource",
    "LangfuseReadClient",
    "LoggedCost",
    "MalformedIdentifierError",
    "ParsedIdentifier",
    "ProviderCost",
    "ProviderCostSource",
    "TierConfig",
    "TierConfigLoadError",
    "TierLevel",
    "TierLevelConfig",
    "TraceSpan",
    "acall_with_retry",
    "acall_with_retry_and_fallback",
    "acall_with_tier_fallback",
    "active_routing_key",
    "add_column_if_missing",
    "arun_agent",
    "build_agent",
    "build_agent_for_level",
    "call_with_retry",
    "call_with_retry_and_fallback",
    "call_with_tier_fallback",
    "create_model",
    "current_session",
    "default_tier_config",
    "flush_tracing",
    "get_provider_for_identifier",
    "get_provider_for_level",
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
    "parse_model_identifier",
    "reconcile",
    "run_additive_migrations",
    "run_agent",
    "setup_langfuse_tracing",
    "start_span",
    "start_trace",
    "timeout_http_client",
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
    if name == "get_provider_for_identifier":
        from . import factory

        return factory.get_provider_for_identifier
    if name == "get_provider_for_level":
        from . import factory

        return factory.get_provider_for_level
    if name == "build_agent_for_level":
        from . import factory

        return factory.build_agent_for_level
    if name == "default_tier_config":
        from . import factory

        return factory.default_tier_config
    if name == "MalformedIdentifierError":
        from . import identifier

        return identifier.MalformedIdentifierError
    if name == "ParsedIdentifier":
        from . import identifier

        return identifier.ParsedIdentifier
    if name == "parse_model_identifier":
        from . import identifier

        return identifier.parse_model_identifier
    if name == "timeout_http_client":
        from . import http

        return http.timeout_http_client
    if name == "AsyncLangfuseReadClient":
        from . import langfuse_async_client

        return langfuse_async_client.AsyncLangfuseReadClient
    if name == "LangfuseReadClient":
        from . import langfuse_client

        return langfuse_client.LangfuseReadClient
    if name == "LangfuseCostLogSource":
        from . import langfuse_cost

        return langfuse_cost.LangfuseCostLogSource
    if name == "LLMProvider":
        from . import provider

        return provider.LLMProvider
    if name == "LEVEL1_DEFAULT":
        from robotsix_llmio.config import tier as _config_tier

        return _config_tier.LEVEL1_DEFAULT
    if name == "LEVEL2_DEFAULT":
        from robotsix_llmio.config import tier as _config_tier

        return _config_tier.LEVEL2_DEFAULT
    if name == "LEVEL3_DEFAULT":
        from robotsix_llmio.config import tier as _config_tier

        return _config_tier.LEVEL3_DEFAULT
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
    if name == "create_model":
        from robotsix_llmio.config import factory as _config_factory

        return _config_factory.create_model
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
    if name == "add_column_if_missing":
        from . import sqlite_utils

        return sqlite_utils.add_column_if_missing
    if name == "arun_agent":
        from . import run

        return run.arun_agent
    if name == "run_additive_migrations":
        from . import sqlite_utils

        return sqlite_utils.run_additive_migrations
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

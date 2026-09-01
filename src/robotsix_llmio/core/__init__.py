"""Provider-agnostic LLM I/O base.

All submodule imports are deferred via PEP 562 ``__getattr__`` so that
importing lightweight helpers (e.g. ``core.retry``) does not eagerly pull in
pydantic-ai or OpenTelemetry at module load time.
"""

from __future__ import annotations

import importlib
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
        DEFAULT_LEVEL1,
        DEFAULT_LEVEL2,
        DEFAULT_LEVEL3,
        FALLBACK_LEVEL1,
        FALLBACK_LEVEL2,
        FALLBACK_LEVEL3,
        VISION_DEFAULT,
        FailoverConfig,
        ProviderSlotConfig,
        TierConfig,
        TierLevel,
        TierLevelConfig,
    )

    from .agent import AgentHandle, build_agent
    from .cost import flush_current_provider, record_cost
    from .cost_log import CostLogSource, CostRecord, CostWindow, LoggedCost
    from .factory import (
        build_agent_for_level,
        default_tier_config,
        get_provider_for_identifier,
        get_provider_for_level,
    )
    from .failover import (
        FailoverStatus,
        ProviderFailoverTracker,
        acall_with_failover,
        call_with_failover,
        get_failover_status,
        get_failover_tracker,
        is_provider_shaped,
        reset_failover_tracker,
    )
    from .http import timeout_http_client
    from .identifier import (
        MalformedIdentifierError,
        ParsedIdentifier,
        parse_model_identifier,
    )
    from .image_tool import build_image_question_tool
    from .langfuse_async_client import AsyncLangfuseReadClient
    from .langfuse_client import LangfuseClientError, LangfuseReadClient
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
        is_usage_exhausted,
    )
    from .run import arun_agent, run_agent
    from .sqlite_utils import (
        add_column_if_missing,
        run_additive_migrations,
        run_multi_table_migrations,
    )
    from .text_utils import html_to_text
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
    "DEFAULT_LEVEL1",
    "DEFAULT_LEVEL2",
    "DEFAULT_LEVEL3",
    "DEFAULT_TOLERANCE",
    "FALLBACK_LEVEL1",
    "FALLBACK_LEVEL2",
    "FALLBACK_LEVEL3",
    "VISION_DEFAULT",
    "AgentHandle",
    "AsyncLangfuseReadClient",
    "CostLogSource",
    "CostRecord",
    "CostWindow",
    "Discrepancy",
    "FailoverConfig",
    "FailoverStatus",
    "LLMProvider",
    "LangfuseClientError",
    "LangfuseCostLogSource",
    "LangfuseReadClient",
    "LoggedCost",
    "MalformedIdentifierError",
    "ParsedIdentifier",
    "ProviderCost",
    "ProviderCostSource",
    "ProviderFailoverTracker",
    "ProviderSlotConfig",
    "TierConfig",
    "TierConfigLoadError",
    "TierLevel",
    "TierLevelConfig",
    "TraceSpan",
    "acall_with_failover",
    "acall_with_retry",
    "acall_with_retry_and_fallback",
    "active_routing_key",
    "add_column_if_missing",
    "arun_agent",
    "build_agent",
    "build_agent_for_level",
    "build_image_question_tool",
    "call_with_failover",
    "call_with_retry",
    "call_with_retry_and_fallback",
    "create_model",
    "current_session",
    "default_tier_config",
    "flush_current_provider",
    "flush_tracing",
    "get_failover_status",
    "get_failover_tracker",
    "get_provider_for_identifier",
    "get_provider_for_level",
    "get_recording_span",
    "get_tracer",
    "html_to_text",
    "install_signal_handlers",
    "is_provider_shaped",
    "is_rate_limited",
    "is_transient",
    "is_usage_exhausted",
    "langfuse_project",
    "langfuse_session",
    "langfuse_trace_url",
    "load_tier_config",
    "make_session_id",
    "parse_model_identifier",
    "reconcile",
    "record_cost",
    "reset_failover_tracker",
    "run_additive_migrations",
    "run_agent",
    "run_multi_table_migrations",
    "setup_langfuse_tracing",
    "start_span",
    "start_trace",
    "timeout_http_client",
]


# Dict-driven PEP 562 lazy imports — each entry maps an export name
# to a (module_path, attr_name) tuple.  Relative paths (".foo") are
# resolved against this package via importlib.import_module.
_SUBMODULE_ATTRS: dict[str, tuple[str, str]] = {
    # -- agent
    "AgentHandle": (".agent", "AgentHandle"),
    "build_agent": (".agent", "build_agent"),
    # -- failover
    "FailoverStatus": (".failover", "FailoverStatus"),
    "ProviderFailoverTracker": (".failover", "ProviderFailoverTracker"),
    "acall_with_failover": (".failover", "acall_with_failover"),
    "call_with_failover": (".failover", "call_with_failover"),
    "get_failover_status": (".failover", "get_failover_status"),
    "get_failover_tracker": (".failover", "get_failover_tracker"),
    "is_provider_shaped": (".failover", "is_provider_shaped"),
    "reset_failover_tracker": (".failover", "reset_failover_tracker"),
    # -- cost_log
    "CostLogSource": (".cost_log", "CostLogSource"),
    "CostRecord": (".cost_log", "CostRecord"),
    "CostWindow": (".cost_log", "CostWindow"),
    "LoggedCost": (".cost_log", "LoggedCost"),
    # -- cost
    "flush_current_provider": (".cost", "flush_current_provider"),
    "record_cost": (".cost", "record_cost"),
    # -- factory
    "get_provider_for_identifier": (".factory", "get_provider_for_identifier"),
    "get_provider_for_level": (".factory", "get_provider_for_level"),
    "build_agent_for_level": (".factory", "build_agent_for_level"),
    "default_tier_config": (".factory", "default_tier_config"),
    # -- identifier
    "MalformedIdentifierError": (".identifier", "MalformedIdentifierError"),
    "ParsedIdentifier": (".identifier", "ParsedIdentifier"),
    "parse_model_identifier": (".identifier", "parse_model_identifier"),
    # -- http
    "timeout_http_client": (".http", "timeout_http_client"),
    # -- image_tool
    "build_image_question_tool": (".image_tool", "build_image_question_tool"),
    # -- langfuse_async_client
    "AsyncLangfuseReadClient": (".langfuse_async_client", "AsyncLangfuseReadClient"),
    # -- langfuse_client
    "LangfuseClientError": (".langfuse_client", "LangfuseClientError"),
    "LangfuseReadClient": (".langfuse_client", "LangfuseReadClient"),
    # -- langfuse_cost
    "LangfuseCostLogSource": (".langfuse_cost", "LangfuseCostLogSource"),
    # -- provider
    "LLMProvider": (".provider", "LLMProvider"),
    # -- provider_cost
    "DEFAULT_TOLERANCE": (".provider_cost", "DEFAULT_TOLERANCE"),
    "Discrepancy": (".provider_cost", "Discrepancy"),
    "ProviderCost": (".provider_cost", "ProviderCost"),
    "ProviderCostSource": (".provider_cost", "ProviderCostSource"),
    "reconcile": (".provider_cost", "reconcile"),
    # -- retry
    "acall_with_retry": (".retry", "acall_with_retry"),
    "acall_with_retry_and_fallback": (".retry", "acall_with_retry_and_fallback"),
    "call_with_retry": (".retry", "call_with_retry"),
    "call_with_retry_and_fallback": (".retry", "call_with_retry_and_fallback"),
    "is_rate_limited": (".retry", "is_rate_limited"),
    "is_transient": (".retry", "is_transient"),
    "is_usage_exhausted": (".retry", "is_usage_exhausted"),
    # -- sqlite_utils
    "add_column_if_missing": (".sqlite_utils", "add_column_if_missing"),
    "run_additive_migrations": (".sqlite_utils", "run_additive_migrations"),
    "run_multi_table_migrations": (".sqlite_utils", "run_multi_table_migrations"),
    # -- run
    "arun_agent": (".run", "arun_agent"),
    "run_agent": (".run", "run_agent"),
    # -- text_utils
    "html_to_text": (".text_utils", "html_to_text"),
    # -- tracing
    "TraceSpan": (".tracing", "TraceSpan"),
    "active_routing_key": (".tracing", "active_routing_key"),
    "current_session": (".tracing", "current_session"),
    "flush_tracing": (".tracing", "flush_tracing"),
    "get_recording_span": (".tracing", "get_recording_span"),
    "get_tracer": (".tracing", "get_tracer"),
    "install_signal_handlers": (".tracing", "install_signal_handlers"),
    "langfuse_project": (".tracing", "langfuse_project"),
    "langfuse_session": (".tracing", "langfuse_session"),
    "langfuse_trace_url": (".tracing", "langfuse_trace_url"),
    "make_session_id": (".tracing", "make_session_id"),
    "setup_langfuse_tracing": (".tracing", "setup_langfuse_tracing"),
    "start_span": (".tracing", "start_span"),
    "start_trace": (".tracing", "start_trace"),
    # -- robotsix_llmio.config (full dotted paths)
    "DEFAULT_LEVEL1": ("robotsix_llmio.config.tier", "DEFAULT_LEVEL1"),
    "DEFAULT_LEVEL2": ("robotsix_llmio.config.tier", "DEFAULT_LEVEL2"),
    "DEFAULT_LEVEL3": ("robotsix_llmio.config.tier", "DEFAULT_LEVEL3"),
    "FALLBACK_LEVEL1": ("robotsix_llmio.config.tier", "FALLBACK_LEVEL1"),
    "FALLBACK_LEVEL2": ("robotsix_llmio.config.tier", "FALLBACK_LEVEL2"),
    "FALLBACK_LEVEL3": ("robotsix_llmio.config.tier", "FALLBACK_LEVEL3"),
    "FailoverConfig": ("robotsix_llmio.config.tier", "FailoverConfig"),
    "VISION_DEFAULT": ("robotsix_llmio.config.tier", "VISION_DEFAULT"),
    "ProviderSlotConfig": ("robotsix_llmio.config.tier", "ProviderSlotConfig"),
    "TierConfig": ("robotsix_llmio.config.tier", "TierConfig"),
    "TierLevel": ("robotsix_llmio.config.tier", "TierLevel"),
    "TierLevelConfig": ("robotsix_llmio.config.tier", "TierLevelConfig"),
    "TierConfigLoadError": ("robotsix_llmio.config.loader", "TierConfigLoadError"),
    "load_tier_config": ("robotsix_llmio.config.loader", "load_tier_config"),
    "create_model": ("robotsix_llmio.config.factory", "create_model"),
}


def __getattr__(name: str) -> Any:  # PEP 562 — lazy heavy imports
    try:
        mod_path, attr = _SUBMODULE_ATTRS[name]
    except KeyError:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}") from None
    mod = importlib.import_module(mod_path, package=__package__)
    return getattr(mod, attr)

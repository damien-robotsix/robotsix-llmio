"""False-positive suppressions for vulture dead-code detection.

This file is parsed as Python by vulture — names referenced at module scope
are treated as "used" and suppress corresponding unused-code warnings.

Do NOT import or execute this file; it is only consumed by vulture's AST
visitor.
"""

# PEP 562 module-level __getattr__ — called by Python's import machinery.
__getattr__

# setup_logging is a public API entry point (called by library consumers).
setup_logging

# Span is imported under TYPE_CHECKING for annotations in _otel.py.
Span

# ---------------------------------------------------------------------------
# claude_sdk/_tool_agent.py
# ---------------------------------------------------------------------------

# Hook callback signature required by the Claude Agent SDK.
context
tool_use_id

# _SdkToolResult — dataclass fields + public methods consumed by callers.
_SdkToolResult.output
_SdkToolResult.all_messages

# _SdkToolAgentHandle — public sync entry point.
_SdkToolAgentHandle.run_sync

# ---------------------------------------------------------------------------
# core/_tracing_processors.py — OpenTelemetry SpanProcessor interface
# ---------------------------------------------------------------------------

# on_start signature mandated by OTel SpanProcessor.
parent_context
_StampProcessor.on_start
_StampProcessor.shutdown

# ---------------------------------------------------------------------------
# core/cost_log.py — dataclass fields / Protocol methods
# ---------------------------------------------------------------------------

CostRecord.id
LoggedCost.record_count
CostLogSource.fetch_logged_cost

# ---------------------------------------------------------------------------
# core/langfuse_async_client.py — public async read methods
# ---------------------------------------------------------------------------

AsyncLangfuseReadClient.fetch_traces_window
AsyncLangfuseReadClient.fetch_trace_detail

# ---------------------------------------------------------------------------
# core/langfuse_client.py — public static helpers
# ---------------------------------------------------------------------------

LangfuseReadClient.parse_timestamp
LangfuseReadClient.observation_provider
LangfuseReadClient.observation_cost

# ---------------------------------------------------------------------------
# core/langfuse_cost.py — public methods of the cost-source adapter
# ---------------------------------------------------------------------------

# fetch_logged_cost is whitelisted once under core/cost_log.py above; vulture
# matches bare names, so the CostLogSource entry already covers this class too.
LangfuseCostLogSource.fetch_logged_cost_by_provider
LangfuseCostLogSource.prune_before

# ---------------------------------------------------------------------------
# core/sqlite_utils.py — Protocol method parameter names
# ---------------------------------------------------------------------------

sql
parameters

# ---------------------------------------------------------------------------
# core/provider_cost.py — Protocol method + Discrepancy dataclass fields
# ---------------------------------------------------------------------------

ProviderCostSource.fetch_provider_cost
Discrepancy.logged_total
Discrepancy.provider_total
Discrepancy.within_tolerance

# ---------------------------------------------------------------------------
# core/tracing.py — signal handler signature (mandated by signal.signal)
# ---------------------------------------------------------------------------

frame
signum

# ---------------------------------------------------------------------------
# openrouter/provider_cost.py — public cost-source methods
# ---------------------------------------------------------------------------

OpenRouterKeyCostSource.fetch_key_usage
# fetch_provider_cost is whitelisted once under core/provider_cost.py above.

# ---------------------------------------------------------------------------
# openrouter/_async_client.py — public async client methods
# ---------------------------------------------------------------------------

AsyncOpenRouterClient.fetch_credits

# ---------------------------------------------------------------------------
# config/tier.py — Pydantic model fields and StrEnum members
# ---------------------------------------------------------------------------

# StrEnum members referenced via TierLevel.LEVEL{N} externally.
TierLevel.LEVEL2
TierLevel.LEVEL3

# Pydantic model fields accessed by pydantic's metaclass machinery, not by
# direct Python name access that vulture would detect. (@model_validator /
# @field_validator methods are handled by ignore_decorators in pyproject.toml.)
TierLevelConfig.provider_kwargs
TierConfig.level1
TierConfig.level2
TierConfig.level3

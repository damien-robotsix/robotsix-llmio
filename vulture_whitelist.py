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
# core/langfuse_client.py — public static helpers
# ---------------------------------------------------------------------------

LangfuseClient.parse_timestamp
LangfuseClient.observation_provider
LangfuseClient.observation_cost

# ---------------------------------------------------------------------------
# core/langfuse_cost.py — public methods of the cost-source adapter
# ---------------------------------------------------------------------------

LangfuseCostSource.fetch_logged_cost
LangfuseCostSource.fetch_logged_cost_by_provider
LangfuseCostSource.prune_before

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
OpenRouterProviderCostSource.fetch_provider_cost

"""Internal OpenTelemetry helpers — not part of the public API.

These exist to eliminate duplicated span-guard boilerplate across modules.
"""

import contextlib
from collections.abc import Iterator
from typing import TYPE_CHECKING, Any

# Semantic-convention operation names (OpenTelemetry GenAI conventions v1.37+)
OP_CHAT = "chat"
OP_EXECUTE_TOOL = "execute_tool"
OP_INVOKE_AGENT = "invoke_agent"

# Semantic-convention attribute keys — gen_ai.* (OpenTelemetry GenAI) and
# langfuse.* (Langfuse). Single source of truth so an upstream rename is a
# one-line change here, not a scattered edit across modules.
GEN_AI_OPERATION_NAME = "gen_ai.operation.name"
GEN_AI_SYSTEM = "gen_ai.system"
GEN_AI_PROVIDER_NAME = "gen_ai.provider.name"
GEN_AI_REQUEST_MODEL = "gen_ai.request.model"
GEN_AI_TOOL_NAME = "gen_ai.tool.name"
GEN_AI_USAGE_COST = "gen_ai.usage.cost"
GEN_AI_USAGE_INPUT_TOKENS = "gen_ai.usage.input_tokens"
GEN_AI_USAGE_OUTPUT_TOKENS = "gen_ai.usage.output_tokens"
GEN_AI_USAGE_CACHE_READ_INPUT_TOKENS = "gen_ai.usage.cache_read_input_tokens"
GEN_AI_USAGE_CACHE_CREATION_INPUT_TOKENS = "gen_ai.usage.cache_creation_input_tokens"
GEN_AI_USAGE_REASONING_TOKENS = "gen_ai.usage.reasoning_tokens"
LANGFUSE_SESSION_ID = "langfuse.session.id"
LANGFUSE_PUBLIC_KEY = "langfuse.public_key"
LANGFUSE_OBSERVATION_INPUT = "langfuse.observation.input"
LANGFUSE_OBSERVATION_OUTPUT = "langfuse.observation.output"
LANGFUSE_OBSERVATION_COST_DETAILS = "langfuse.observation.cost_details"
LANGFUSE_COST_DETAILS_TOTAL_KEY = "total"
LANGFUSE_OBSERVATION_METADATA_PROVIDER = "langfuse.observation.metadata.provider"

_DEFAULT_LANGFUSE_BASE_URL = "https://cloud.langfuse.com"

if TYPE_CHECKING:
    from opentelemetry.trace import Span


def get_recording_span() -> "Span | None":
    """Return the current OTel span if recording, else None. No-op without OTel."""
    try:
        from opentelemetry import trace as otel_trace  # type: ignore[import-untyped]
    except ImportError:
        return None
    span = otel_trace.get_current_span()
    return span if (span is not None and span.is_recording()) else None


def get_tracer(name: str) -> Any | None:
    """Return an OTel tracer for *name*, or ``None`` when OpenTelemetry is not
    installed. The tracer late-binds to whatever provider is active when a span
    is started, so it's safe to fetch before tracing is configured."""
    try:
        from opentelemetry import trace as otel_trace  # type: ignore[import-untyped]
    except ImportError:
        return None
    return otel_trace.get_tracer(name)


@contextlib.contextmanager
def start_span(
    tracer: Any | None, name: str, attributes: dict[str, Any] | None = None
) -> Iterator[Any | None]:
    """Start a current span on *tracer*, set *attributes* (skipping ``None``
    values), and yield it. A no-op yielding ``None`` when *tracer* is ``None``
    (OTel absent); a non-recording span when no SDK provider is configured."""
    if tracer is None:
        yield None
        return
    with tracer.start_as_current_span(name) as span:
        for key, value in (attributes or {}).items():
            if value is not None:
                span.set_attribute(key, value)
        yield span

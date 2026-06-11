"""Pin the semantic-convention constants to their wire strings.

``core/_otel.py`` owns the single source of truth for every OpenTelemetry
GenAI / Langfuse attribute-key (and operation-name) literal emitted on spans.
The constants are a refactor convenience, but the *values* are a wire contract:
a typo in a constant would silently change an emitted attribute name. These
tests lock each constant to its exact expected string so such a typo is caught.
"""

from __future__ import annotations

from robotsix_llmio.core import _otel

EXPECTED: dict[str, str] = {
    # Operation-name values.
    "OP_CHAT": "chat",
    "OP_EXECUTE_TOOL": "execute_tool",
    "OP_INVOKE_AGENT": "invoke_agent",
    # gen_ai.* attribute keys.
    "GEN_AI_OPERATION_NAME": "gen_ai.operation.name",
    "GEN_AI_SYSTEM": "gen_ai.system",
    "GEN_AI_PROVIDER_NAME": "gen_ai.provider.name",
    "GEN_AI_REQUEST_MODEL": "gen_ai.request.model",
    "GEN_AI_TOOL_NAME": "gen_ai.tool.name",
    "GEN_AI_USAGE_COST": "gen_ai.usage.cost",
    "GEN_AI_USAGE_INPUT_TOKENS": "gen_ai.usage.input_tokens",
    "GEN_AI_USAGE_OUTPUT_TOKENS": "gen_ai.usage.output_tokens",
    "GEN_AI_USAGE_CACHE_READ_INPUT_TOKENS": "gen_ai.usage.cache_read_input_tokens",
    "GEN_AI_USAGE_CACHE_CREATION_INPUT_TOKENS": (
        "gen_ai.usage.cache_creation_input_tokens"
    ),
    "GEN_AI_USAGE_REASONING_TOKENS": "gen_ai.usage.reasoning_tokens",
    # langfuse.* attribute keys.
    "LANGFUSE_SESSION_ID": "langfuse.session.id",
    "LANGFUSE_PUBLIC_KEY": "langfuse.public_key",
    "LANGFUSE_OBSERVATION_INPUT": "langfuse.observation.input",
    "LANGFUSE_OBSERVATION_OUTPUT": "langfuse.observation.output",
    "LANGFUSE_OBSERVATION_COST_DETAILS": "langfuse.observation.cost_details",
    "LANGFUSE_COST_DETAILS_TOTAL_KEY": "total",
    "LANGFUSE_OBSERVATION_METADATA_PROVIDER": "langfuse.observation.metadata.provider",
}


def test_otel_constants_match_wire_strings() -> None:
    for name, value in EXPECTED.items():
        assert getattr(_otel, name) == value, name

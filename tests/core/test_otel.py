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
    "LANGFUSE_OBSERVATION_METADATA_REASONING": (
        "langfuse.observation.metadata.reasoning"
    ),
    # Langfuse OTLP wire path.
    "_LANGFUSE_OTEL_TRACES_PATH": "/api/public/otel/v1/traces",
    # Environment-variable names (consumed by tracing.py for credentials).
    "ENV_LANGFUSE_PUBLIC_KEY": "LANGFUSE_PUBLIC_KEY",
    "ENV_LANGFUSE_SECRET_KEY": "LANGFUSE_SECRET_KEY",
    "ENV_LANGFUSE_BASE_URL": "LANGFUSE_BASE_URL",
    "ENV_LANGFUSE_PROJECT_ID": "LANGFUSE_PROJECT_ID",
}


def test_otel_constants_match_wire_strings() -> None:
    for name, value in EXPECTED.items():
        assert getattr(_otel, name) == value, name


def test_env_example_documents_langfuse_live_test_credentials() -> None:
    """The Langfuse credentials the live tracing tests consume must be listed
    in ``tests/.env.example`` so contributors can discover what to set.

    ``tests/.env.example`` documents only the credentials the opt-in live test
    suite needs (the library's full runtime env interface lives in the README),
    so ``LANGFUSE_PROJECT_ID`` — read at runtime but not by any test — is
    intentionally excluded here."""
    import pathlib

    env_constants = {
        _otel.ENV_LANGFUSE_PUBLIC_KEY,
        _otel.ENV_LANGFUSE_SECRET_KEY,
        _otel.ENV_LANGFUSE_BASE_URL,
    }
    env_example_path = pathlib.Path(__file__).parents[1] / ".env.example"
    env_example = env_example_path.read_text()
    for name in env_constants:
        assert f"{name}=" in env_example, f"{name} not found in {env_example_path}"

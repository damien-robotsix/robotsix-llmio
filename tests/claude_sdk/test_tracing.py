"""Tracing tests for the Claude Agent SDK transport — span attributes,
token usage, reasoning metadata, and no-tools request spans."""

from __future__ import annotations

import asyncio
import json

import pytest

pytest.importorskip("pydantic_ai")

from pydantic_ai.messages import (
    ModelRequest,
    UserPromptPart,
)
from pydantic_ai.models import ModelRequestParameters
from pydantic_ai.tools import Tool as PydanticTool

from robotsix_llmio.claude_sdk._chat_messages import _chat_messages_input
from robotsix_llmio.claude_sdk.model import ClaudeSDKModel
from robotsix_llmio.claude_sdk.provider import ClaudeSDKProvider
from robotsix_llmio.core.agent import AgentHandle
from robotsix_llmio.core.tracing import (
    GEN_AI_OPERATION_NAME,
    GEN_AI_PROVIDER_NAME,
    GEN_AI_REQUEST_MODEL,
    GEN_AI_SYSTEM,
    GEN_AI_USAGE_INPUT_TOKENS,
    GEN_AI_USAGE_OUTPUT_TOKENS,
    LANGFUSE_OBSERVATION_INPUT,
    LANGFUSE_OBSERVATION_METADATA_REASONING,
    LANGFUSE_OBSERVATION_OUTPUT,
    OP_CHAT,
    OP_INVOKE_AGENT,
)

from .conftest import (
    _HAIKU_AT_LEVEL1,
    _echo_sync,
    _FakeThinkingBlock,
    _install_fake_sdk,
)

# ---------------------------------------------------------------------------
# Tracing: the system prompt (sent to the SDK) is surfaced on the generation
# ---------------------------------------------------------------------------


def test_chat_messages_input_renders_system_and_user():
    raw = _chat_messages_input("be terse", "hello there")
    msgs = json.loads(raw)
    assert msgs == [
        {"role": "system", "content": "be terse"},
        {"role": "user", "content": "hello there"},
    ]


def test_generation_span_input_includes_system_prompt(
    monkeypatch, otel_exporter_tracer
):
    """End-to-end: the ``chat`` generation span records system + user as chat
    messages, so the system prompt is visible in the trace (not just input)."""
    exporter, tracer = otel_exporter_tracer
    import robotsix_llmio.claude_sdk._tool_agent as _ta

    # Route the module's spans to our isolated, recording provider (the offline
    # suite installs no global TracerProvider).
    monkeypatch.setattr(_ta, "get_tracer", lambda _name: tracer)

    fake = _install_fake_sdk(monkeypatch)

    async def _fake_query(*, prompt, options):
        yield fake.AssistantMessage("the answer")
        yield fake.ResultMessage({"input_tokens": 1, "output_tokens": 1})

    fake.query = _fake_query

    handle = ClaudeSDKProvider().build_agent(
        level=1,
        tier_config=_HAIKU_AT_LEVEL1,
        system_prompt="SYS_MARKER stay precise",
        tools=[PydanticTool(_echo_sync, name="echo_sync")],
    )
    handle.run_sync("USER_MARKER hi")
    handle.close()

    spans = exporter.get_finished_spans()

    def _input_messages(predicate) -> list:
        matched = [s for s in spans if predicate(s)]
        assert matched, f"no matching span in {[s.name for s in spans]}"
        return json.loads(matched[0].attributes[LANGFUSE_OBSERVATION_INPUT])

    # The child generation span carries system + user...
    chat = _input_messages(lambda s: s.name.startswith("chat "))
    assert chat[0]["role"] == "system" and "SYS_MARKER" in chat[0]["content"]
    assert chat[1]["role"] == "user" and "USER_MARKER" in chat[1]["content"]

    # ...and so does the root agent-run span (which becomes the trace), so the
    # system prompt is visible at the trace root, not only on the generation.
    root = _input_messages(
        lambda s: s.attributes.get(GEN_AI_OPERATION_NAME) == OP_INVOKE_AGENT
    )
    assert root[0]["role"] == "system" and "SYS_MARKER" in root[0]["content"]
    assert root[1]["role"] == "user" and "USER_MARKER" in root[1]["content"]


def test_spans_set_gen_ai_provider_name(monkeypatch, otel_exporter_tracer):
    """Both the ``chat`` generation span and the root agent-run span stamp
    ``gen_ai.provider.name`` with the transport provider, matching the
    OpenRouter transport's span attributes."""
    exporter, tracer = otel_exporter_tracer
    import robotsix_llmio.claude_sdk._tool_agent as _ta
    from robotsix_llmio.claude_sdk.model import PROVIDER_NAME

    monkeypatch.setattr(_ta, "get_tracer", lambda _name: tracer)

    fake = _install_fake_sdk(monkeypatch)

    async def _fake_query(*, prompt, options):
        yield fake.AssistantMessage("the answer")
        yield fake.ResultMessage({"input_tokens": 1, "output_tokens": 1})

    fake.query = _fake_query

    handle = ClaudeSDKProvider().build_agent(
        level=1,
        tier_config=_HAIKU_AT_LEVEL1,
        system_prompt="be precise",
        tools=[PydanticTool(_echo_sync, name="echo_sync")],
    )
    handle.run_sync("hi")
    handle.close()

    spans = exporter.get_finished_spans()

    chat = next((s for s in spans if s.name.startswith("chat ")), None)
    assert chat is not None, f"no chat span in {[s.name for s in spans]}"
    assert chat.attributes.get(GEN_AI_PROVIDER_NAME) == PROVIDER_NAME

    root = next(
        (
            s
            for s in spans
            if s.attributes.get(GEN_AI_OPERATION_NAME) == OP_INVOKE_AGENT
        ),
        None,
    )
    assert root is not None, f"no root span in {[s.name for s in spans]}"
    assert root.attributes.get(GEN_AI_PROVIDER_NAME) == PROVIDER_NAME


def test_tool_path_stamps_token_usage_on_generation_span(
    monkeypatch, otel_exporter_tracer
):
    """The ``chat`` generation span carries non-zero token usage from the SDK
    result, so Langfuse reports ``totalTokens`` and ``usageDetails``."""
    exporter, tracer = otel_exporter_tracer
    import robotsix_llmio.claude_sdk._tool_agent as _ta

    monkeypatch.setattr(_ta, "get_tracer", lambda _name: tracer)

    fake = _install_fake_sdk(monkeypatch)

    async def _fake_query(*, prompt, options):
        yield fake.AssistantMessage("substantive answer")
        yield fake.ResultMessage({"input_tokens": 42, "output_tokens": 17})

    fake.query = _fake_query

    handle = ClaudeSDKProvider().build_agent(
        level=1,
        tier_config=_HAIKU_AT_LEVEL1,
        system_prompt="be precise",
        tools=[PydanticTool(_echo_sync, name="echo_sync")],
    )
    handle.run_sync("hi")
    handle.close()

    spans = exporter.get_finished_spans()
    chat = next((s for s in spans if s.name.startswith("chat ")), None)
    assert chat is not None, f"no chat span in {[s.name for s in spans]}"

    # Core invariant: when content was produced, token usage MUST be non-zero.
    output = chat.attributes.get(LANGFUSE_OBSERVATION_OUTPUT)
    assert output and output.strip(), "chat span has no output"
    in_tok = chat.attributes.get(GEN_AI_USAGE_INPUT_TOKENS)
    out_tok = chat.attributes.get(GEN_AI_USAGE_OUTPUT_TOKENS)
    assert in_tok == 42, f"expected input_tokens=42, got {in_tok}"
    assert out_tok == 17, f"expected output_tokens=17, got {out_tok}"


def test_tool_path_prefers_model_usage_over_usage(monkeypatch, otel_exporter_tracer):
    """When the SDK result has token data in ``model_usage`` rather than
    ``usage``, the generation span still records non-zero tokens."""
    exporter, tracer = otel_exporter_tracer
    import robotsix_llmio.claude_sdk._tool_agent as _ta

    monkeypatch.setattr(_ta, "get_tracer", lambda _name: tracer)

    fake = _install_fake_sdk(monkeypatch)

    async def _fake_query(*, prompt, options):
        yield fake.AssistantMessage("substantive answer")
        yield fake.ResultMessage(model_usage={"input_tokens": 55, "output_tokens": 23})

    fake.query = _fake_query

    handle = ClaudeSDKProvider().build_agent(
        level=1,
        tier_config=_HAIKU_AT_LEVEL1,
        system_prompt="be precise",
        tools=[PydanticTool(_echo_sync, name="echo_sync")],
    )
    handle.run_sync("hi")
    handle.close()

    spans = exporter.get_finished_spans()
    chat = next((s for s in spans if s.name.startswith("chat ")), None)
    assert chat is not None, f"no chat span in {[s.name for s in spans]}"
    assert chat.attributes.get(GEN_AI_USAGE_INPUT_TOKENS) == 55
    assert chat.attributes.get(GEN_AI_USAGE_OUTPUT_TOKENS) == 23


# --- no-tools request() span attributes (regression) -----------------------


def test_notools_request_stamps_gen_ai_attributes(monkeypatch, otel_exporter_tracer):
    """After ``ClaudeSDKModel.request()`` runs with a recording span active, that
    span carries provider.model identity, system, and token-usage attributes
    — independently of whether cost was recorded, and consistent with the
    tool-loop path."""
    exporter, tracer = otel_exporter_tracer

    fake = _install_fake_sdk(monkeypatch)

    async def _fake_query(*, prompt, options):
        yield fake.AssistantMessage("hello world")
        yield fake.ResultMessage({"input_tokens": 10, "output_tokens": 5})

    fake.query = _fake_query

    model = ClaudeSDKModel("opus")

    async def _run():
        with tracer.start_as_current_span("root"):
            return await model.request(
                [ModelRequest(parts=[UserPromptPart(content="hi")])],
                model_settings=None,
                model_request_parameters=ModelRequestParameters(output_mode="text"),
            )

    asyncio.run(_run())

    spans = exporter.get_finished_spans()
    assert len(spans) == 1, f"expected 1 span, got {[s.name for s in spans]}"
    attrs = spans[0].attributes
    assert attrs[GEN_AI_OPERATION_NAME] == OP_CHAT
    assert attrs[GEN_AI_PROVIDER_NAME] == "claude-sdk"
    assert attrs[GEN_AI_SYSTEM] == "anthropic"
    assert attrs[GEN_AI_REQUEST_MODEL] == "opus"
    assert attrs[GEN_AI_USAGE_INPUT_TOKENS] == 10
    assert attrs[GEN_AI_USAGE_OUTPUT_TOKENS] == 5


def test_notools_request_stamps_tokens_from_model_usage(
    monkeypatch,
    otel_exporter_tracer,
):
    """When the SDK result carries token data in ``model_usage`` rather than
    ``usage``, the generation span still records non-zero tokens."""
    exporter, tracer = otel_exporter_tracer

    fake = _install_fake_sdk(monkeypatch)

    async def _fake_query(*, prompt, options):
        yield fake.AssistantMessage("hello world")
        yield fake.ResultMessage(model_usage={"input_tokens": 30, "output_tokens": 12})

    fake.query = _fake_query

    model = ClaudeSDKModel("opus")

    async def _run():
        with tracer.start_as_current_span("root"):
            return await model.request(
                [ModelRequest(parts=[UserPromptPart(content="hi")])],
                model_settings=None,
                model_request_parameters=ModelRequestParameters(output_mode="text"),
            )

    resp = asyncio.run(_run())
    # The pydantic-ai usage should also reflect model_usage tokens.
    assert resp.usage.input_tokens == 30
    assert resp.usage.output_tokens == 12

    spans = exporter.get_finished_spans()
    assert len(spans) == 1, f"expected 1 span, got {[s.name for s in spans]}"
    attrs = spans[0].attributes
    assert attrs[GEN_AI_USAGE_INPUT_TOKENS] == 30
    assert attrs[GEN_AI_USAGE_OUTPUT_TOKENS] == 12


def test_notools_request_skips_missing_usage(monkeypatch, otel_exporter_tracer):
    """When the SDK ResultMessage has no usage dict, token attributes are not set
    (no spurious zeros), but identity attributes are still stamped."""
    exporter, tracer = otel_exporter_tracer

    fake = _install_fake_sdk(monkeypatch)

    async def _fake_query(*, prompt, options):
        yield fake.AssistantMessage("hello")
        yield fake.ResultMessage()  # no usage

    fake.query = _fake_query

    model = ClaudeSDKModel("sonnet")

    async def _run():
        with tracer.start_as_current_span("root"):
            return await model.request(
                [ModelRequest(parts=[UserPromptPart(content="hi")])],
                model_settings=None,
                model_request_parameters=ModelRequestParameters(output_mode="text"),
            )

    asyncio.run(_run())

    spans = exporter.get_finished_spans()
    assert len(spans) == 1
    attrs = spans[0].attributes
    # Identity always set.
    assert attrs[GEN_AI_PROVIDER_NAME] == "claude-sdk"
    assert attrs[GEN_AI_SYSTEM] == "anthropic"
    assert attrs[GEN_AI_REQUEST_MODEL] == "sonnet"
    # Token keys absent entirely.
    assert GEN_AI_USAGE_INPUT_TOKENS not in attrs
    assert GEN_AI_USAGE_OUTPUT_TOKENS not in attrs


def test_no_span_recording_is_noop(monkeypatch):
    """When no span is recording (OpenTelemetry absent or no TracerProvider),
    ``request()`` behaves exactly as before — no exception, no writes."""
    fake = _install_fake_sdk(monkeypatch)

    async def _fake_query(*, prompt, options):
        yield fake.AssistantMessage("ok")
        yield fake.ResultMessage({"input_tokens": 1, "output_tokens": 1})

    fake.query = _fake_query

    model = ClaudeSDKModel("haiku")
    # No OTel tracer installed — get_recording_span() returns None.
    response = asyncio.run(
        model.request(
            [ModelRequest(parts=[UserPromptPart(content="hi")])],
            model_settings=None,
            model_request_parameters=ModelRequestParameters(output_mode="text"),
        )
    )
    assert response.model_name == "haiku"
    assert response.parts[0].content == "ok"


# --- extended-thinking reasoning surfaced in traces ------------------------


def test_tool_path_records_reasoning_on_generation_span(monkeypatch):
    """A ``ThinkingBlock`` in the tool-loop stream lands as reasoning metadata
    on the ``chat`` generation span (so Langfuse shows the model's reasoning,
    not just the answer and tool calls)."""
    pytest.importorskip("opentelemetry.sdk")
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import SimpleSpanProcessor
    from opentelemetry.sdk.trace.export.in_memory_span_exporter import (
        InMemorySpanExporter,
    )

    import robotsix_llmio.claude_sdk._tool_agent as _ta

    exporter = InMemorySpanExporter()
    provider_obj = TracerProvider()
    provider_obj.add_span_processor(SimpleSpanProcessor(exporter))
    tracer = provider_obj.get_tracer("test")
    monkeypatch.setattr(_ta, "get_tracer", lambda _name: tracer)

    fake = _install_fake_sdk(monkeypatch)

    async def _fake_query(*, prompt, options):
        msg = fake.AssistantMessage("the answer")
        msg.content = [
            _FakeThinkingBlock("weighing the options"),
            fake.TextBlock("the answer"),
        ]
        yield msg
        yield fake.ResultMessage({"input_tokens": 1, "output_tokens": 1})

    fake.query = _fake_query

    handle = ClaudeSDKProvider().build_agent(
        level=1,
        tier_config=_HAIKU_AT_LEVEL1,
        system_prompt="be precise",
        tools=[PydanticTool(_echo_sync, name="echo_sync")],
    )
    handle.run_sync("hi")
    handle.close()

    spans = exporter.get_finished_spans()
    chat = next((s for s in spans if s.name.startswith("chat ")), None)
    assert chat is not None, f"no chat span in {[s.name for s in spans]}"
    assert (
        chat.attributes.get(LANGFUSE_OBSERVATION_METADATA_REASONING)
        == "weighing the options"
    )


def test_notools_request_records_reasoning_metadata(monkeypatch):
    """A ``ThinkingBlock`` in the no-tools stream lands as reasoning metadata on
    the active generation span."""
    pytest.importorskip("opentelemetry.sdk")
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import SimpleSpanProcessor
    from opentelemetry.sdk.trace.export.in_memory_span_exporter import (
        InMemorySpanExporter,
    )

    fake = _install_fake_sdk(monkeypatch)

    async def _fake_query(*, prompt, options):
        msg = fake.AssistantMessage("hello world")
        msg.content = [
            _FakeThinkingBlock("planning the reply"),
            fake.TextBlock("hello world"),
        ]
        yield msg
        yield fake.ResultMessage({"input_tokens": 10, "output_tokens": 5})

    fake.query = _fake_query

    exporter = InMemorySpanExporter()
    provider_obj = TracerProvider()
    provider_obj.add_span_processor(SimpleSpanProcessor(exporter))
    tracer = provider_obj.get_tracer("test")

    model = ClaudeSDKModel("opus")

    async def _run():
        with tracer.start_as_current_span("root"):
            return await model.request(
                [ModelRequest(parts=[UserPromptPart(content="hi")])],
                model_settings=None,
                model_request_parameters=ModelRequestParameters(output_mode="text"),
            )

    asyncio.run(_run())

    spans = exporter.get_finished_spans()
    assert len(spans) == 1
    assert (
        spans[0].attributes.get(LANGFUSE_OBSERVATION_METADATA_REASONING)
        == "planning the reply"
    )


def test_notools_path_returns_agent_handle():
    """When *tools* is None/empty, ``build_agent`` returns a standard
    ``AgentHandle`` wrapping a pydantic-ai ``Agent`` — the existing
    no-tools path is unchanged."""
    provider = ClaudeSDKProvider()
    handle = provider.build_agent(
        level=1,
        tier_config=_HAIKU_AT_LEVEL1,
        system_prompt="You are helpful.",
        tools=None,
    )
    # With no tools the super().build_agent() path wraps a pydantic-ai Agent.
    assert isinstance(handle, AgentHandle)
    assert handle._agent is not None  # type: ignore[attr-defined]
    handle.close()


def test_tools_empty_list_also_returns_agent_handle():
    """Empty tools list is falsy → delegates to the no-tools AgentHandle path."""
    provider = ClaudeSDKProvider()
    handle = provider.build_agent(
        level=1,
        tier_config=_HAIKU_AT_LEVEL1,
        system_prompt="You are helpful.",
        tools=[],
    )
    assert isinstance(handle, AgentHandle)
    handle.close()

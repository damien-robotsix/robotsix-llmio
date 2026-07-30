"""Tracing tests for the Claude Agent SDK transport — span attributes,
token usage, reasoning metadata, and no-tools request spans."""

from __future__ import annotations

import json

import pytest

pytest.importorskip("pydantic_ai")

from pydantic_ai.tools import Tool as PydanticTool

from robotsix_llmio.claude_sdk._chat_messages import _chat_messages_input
from robotsix_llmio.claude_sdk.provider import ClaudeSDKProvider
from robotsix_llmio.core.tracing import (
    GEN_AI_OPERATION_NAME,
    GEN_AI_PROVIDER_NAME,
    GEN_AI_USAGE_INPUT_TOKENS,
    GEN_AI_USAGE_OUTPUT_TOKENS,
    LANGFUSE_OBSERVATION_INPUT,
    LANGFUSE_OBSERVATION_OUTPUT,
    OP_INVOKE_AGENT,
)

from .conftest import (
    _HAIKU_AT_LEVEL1,
    _echo_sync,
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

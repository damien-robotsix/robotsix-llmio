"""Tracing tests for extended-thinking reasoning metadata surfaced on
generation spans, plus AgentHandle return-type edge cases."""

from __future__ import annotations

import asyncio

import pytest

pytest.importorskip("pydantic_ai")

from pydantic_ai.messages import (
    ModelRequest,
    UserPromptPart,
)
from pydantic_ai.models import ModelRequestParameters
from pydantic_ai.tools import Tool as PydanticTool

from robotsix_llmio.claude_sdk.model import ClaudeSDKModel
from robotsix_llmio.claude_sdk.provider import ClaudeSDKProvider
from robotsix_llmio.core.agent import AgentHandle
from robotsix_llmio.core.tracing import (
    LANGFUSE_OBSERVATION_METADATA_REASONING,
)

from .conftest import (
    _HAIKU_AT_LEVEL1,
    _echo_sync,
    _FakeThinkingBlock,
    _install_fake_sdk,
)

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

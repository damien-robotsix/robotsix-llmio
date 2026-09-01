"""Shared fixtures for claude_sdk tests."""

from __future__ import annotations

import sys
from types import SimpleNamespace
from typing import Any

import pytest
from pydantic import BaseModel as _BM
from pydantic_ai.tools import Tool as PydanticTool

from robotsix_llmio.claude_sdk._tool_agent import _SdkToolAgentHandle
from robotsix_llmio.claude_sdk.provider import ClaudeSDKProvider
from robotsix_llmio.config.tier import (
    ProviderSlotConfig,
    TierConfig,
    TierLevelConfig,
)

# ---------------------------------------------------------------------------
# Shared TierConfig constants
# ---------------------------------------------------------------------------

# All three levels are required by ProviderSlotConfig; the levels a test does
# not exercise are filled with the same claudeSDK models the baked default
# slot uses.
_CLAUDE_SLOT = ProviderSlotConfig(
    level1=TierLevelConfig(model="claudeSDK-haiku"),
    level2=TierLevelConfig(model="claudeSDK-opus"),
    level3=TierLevelConfig(model="claudeSDK-claude-fable-5"),
)
_HAIKU_AT_LEVEL1 = TierConfig(default=_CLAUDE_SLOT)
_OPUS_AT_LEVEL2 = TierConfig(default=_CLAUDE_SLOT)


# ---------------------------------------------------------------------------
# Shared fake SDK module (offline tests)
# ---------------------------------------------------------------------------


def _fake_sdk_module() -> SimpleNamespace:
    """Return a fake ``claude_agent_sdk`` namespace for offline tests."""
    tool_regs: list[dict[str, Any]] = []
    server_calls: list[dict[str, Any]] = []

    def _fake_tool(
        name: str, description: str | None, parameters_json_schema: dict[str, Any]
    ):
        tool_regs.append(
            {"name": name, "description": description, "schema": parameters_json_schema}
        )

        def _decorator(fn):
            return fn

        return _decorator

    def _fake_create_sdk_mcp_server(name: str, tools: list):
        server_calls.append({"name": name, "tools": list(tools)})
        return SimpleNamespace()

    class _FakeTextBlock:
        def __init__(self, text: str) -> None:
            self.text = text

    class _FakeAssistantMessage:
        def __init__(self, text: str) -> None:
            self.content = [_FakeTextBlock(text)]

    class _FakeResultMessage:
        def __init__(
            self,
            usage: dict[str, int] | None = None,
            *,
            model_usage: dict[str, int] | None = None,
        ) -> None:
            self.usage = usage
            self.model_usage = model_usage
            self.result = None
            self.total_cost_usd = None
            self.session_id = "fake-session"

    class _FakeClaudeAgentOptions:
        def __init__(self, **kwargs: Any) -> None:
            for k, v in kwargs.items():
                setattr(self, k, v)

    ns = SimpleNamespace()
    ns.tool = _fake_tool
    ns.create_sdk_mcp_server = _fake_create_sdk_mcp_server
    ns.TextBlock = _FakeTextBlock
    ns.AssistantMessage = _FakeAssistantMessage
    ns.ResultMessage = _FakeResultMessage
    ns.ClaudeAgentOptions = _FakeClaudeAgentOptions
    # Attach record-keeping
    ns._tool_regs = tool_regs
    ns._server_calls = server_calls
    return ns


def _install_fake_sdk(monkeypatch) -> SimpleNamespace:
    """Install a fake ``claude_agent_sdk`` module and return its namespace."""
    fake = _fake_sdk_module()
    # Install via monkeypatch ONLY: a preceding raw
    # `sys.modules["claude_agent_sdk"] = fake` would make monkeypatch record
    # the fake as the "original" and restore it (not remove it) on teardown,
    # leaking the incomplete stub into later tests (e.g. test_confine_hook's
    # `from claude_agent_sdk import create_sdk_mcp_server`).
    monkeypatch.setitem(sys.modules, "claude_agent_sdk", fake)
    return fake


# ---------------------------------------------------------------------------
# Shared test tool
# ---------------------------------------------------------------------------


def _echo_sync(text: str) -> str:
    """Echo the input."""
    return text


# ---------------------------------------------------------------------------
# Shared helpers for run_sync / run tests
# ---------------------------------------------------------------------------


def _capturing_query(fake, captured: dict):
    """A fake SDK ``query`` that records the prompt it was handed."""

    async def _q(*, prompt, options):
        captured["prompt"] = prompt
        yield fake.AssistantMessage("done")
        yield fake.ResultMessage({"input_tokens": 1, "output_tokens": 1})

    return _q


def _tool_handle():
    handle = ClaudeSDKProvider().build_agent(
        level=1,
        tier_config=_HAIKU_AT_LEVEL1,
        system_prompt="sys",
        tools=[PydanticTool(_echo_sync, name="echo_sync")],
    )
    return handle


# ---------------------------------------------------------------------------
# Shared pydantic models for output tests
# ---------------------------------------------------------------------------


class _Verdict(_BM):
    verdict: str
    auto_merge_eligible: bool = False


class _AltVerdict(_BM):
    outcome: str


def _make_minimal_handle(
    output_type: Any = str,
    sdk_model: str = "opus",
    system_prompt: str = "sys",
    server: Any = None,
    allowed_tools: list[str] | None = None,
) -> _SdkToolAgentHandle:
    """Construct ``_SdkToolAgentHandle`` with stub arguments for unit tests."""
    return _SdkToolAgentHandle(
        sdk_model=sdk_model,
        system_prompt=system_prompt,
        server=server,
        allowed_tools=allowed_tools or [],
        output_type=output_type,
    )


# ---------------------------------------------------------------------------
# Shared fake thinking block for extended-thinking / reasoning tests
# ---------------------------------------------------------------------------


class _FakeThinkingBlock:
    """A fake SDK thinking block — matched by class name in ``_stream_query``."""

    def __init__(self, thinking: str) -> None:
        self.thinking = thinking


_FakeThinkingBlock.__name__ = "ThinkingBlock"


# ---------------------------------------------------------------------------
# Shared OTel fixture
# ---------------------------------------------------------------------------


@pytest.fixture
def otel_exporter_tracer():
    """Set up an isolated OTel recording provider that routes spans to an
    InMemorySpanExporter so tests can inspect finished spans offline.
    Yields ``(exporter, tracer)``."""
    pytest.importorskip("opentelemetry.sdk")
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import SimpleSpanProcessor
    from opentelemetry.sdk.trace.export.in_memory_span_exporter import (
        InMemorySpanExporter,
    )

    exporter = InMemorySpanExporter()
    provider_obj = TracerProvider()
    provider_obj.add_span_processor(SimpleSpanProcessor(exporter))
    tracer = provider_obj.get_tracer("test")
    yield exporter, tracer

"""Claude Agent SDK transport — prompt rendering, usage mapping, guards,
transient, and tool-loop bridge.  Offline only: the live single-turn and
tool round-trip tests are exercised separately (need the ``claude`` CLI
+ login)."""

from __future__ import annotations

import asyncio
import json
import logging
import sys
from types import SimpleNamespace
from typing import Any, ClassVar

import pytest
from pydantic import BaseModel as _BM

pytest.importorskip("pydantic_ai")

from pydantic_ai.exceptions import UserError
from pydantic_ai.messages import (
    ModelRequest,
    ModelResponse,
    RetryPromptPart,
    SystemPromptPart,
    TextPart,
    ToolReturnPart,
    UserPromptPart,
)
from pydantic_ai.models import ModelRequestParameters
from pydantic_ai.tools import RunContext
from pydantic_ai.tools import Tool as PydanticTool

from robotsix_llmio.claude_sdk._tool_agent import (
    _chat_messages_input,
    _convert_tools,
    _extract_json_object,
    _parse_output,
    _SdkToolAgentHandle,
    _SdkToolResult,
)
from robotsix_llmio.claude_sdk._usage import map_usage_dict
from robotsix_llmio.claude_sdk.model import (
    ClaudeSDKModel,
    _map_usage,
    render_prompt,
)
from robotsix_llmio.claude_sdk.provider import (
    ClaudeSDKProvider,
)
from robotsix_llmio.claude_sdk.transient import (
    is_claude_sdk_transient,
    is_claude_sdk_turn_limit,
)
from robotsix_llmio.config.tier import (
    TierConfig,
    TierLevelConfig,
)
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

# Inline TierConfig for tests that need a specific model at a given level.
_HAIKU_AT_LEVEL1 = TierConfig(
    level1=TierLevelConfig(model="claudeSDK-haiku"),
)
_OPUS_AT_LEVEL2 = TierConfig(
    level1=TierLevelConfig(model="claudeSDK-haiku"),
    level2=TierLevelConfig(model="claudeSDK-opus"),
)

# --- prompt rendering ------------------------------------------------------


def test_single_user_turn_sent_verbatim():
    msgs = [ModelRequest(parts=[UserPromptPart(content="hello there")])]
    assert render_prompt(msgs) == "hello there"


def test_multi_turn_rendered_as_transcript():
    msgs = [
        ModelRequest(parts=[UserPromptPart(content="first")]),
        ModelResponse(parts=[TextPart(content="bad json")]),
        ModelRequest(parts=[RetryPromptPart(content="invalid, retry", tool_name=None)]),
    ]
    out = render_prompt(msgs)
    assert "User: first" in out
    assert "Assistant: bad json" in out
    assert "User:" in out.split("Assistant: bad json")[1]  # retry rendered last


def test_tool_return_part_rendered_as_user_text():
    msgs = [
        ModelRequest(
            parts=[ToolReturnPart(tool_name="lookup", content="42", tool_call_id="c1")]
        )
    ]
    assert "Tool result (lookup): 42" in render_prompt(msgs)


# --- system prompt assembly ------------------------------------------------


def _params(output_mode="text"):
    return ModelRequestParameters(output_mode=output_mode)


def test_system_text_combines_instructions_and_system_parts():
    m = ClaudeSDKModel("opus")
    msgs = [
        ModelRequest(
            parts=[
                SystemPromptPart(content="be terse"),
                UserPromptPart(content="hi"),
            ],
            instructions="answer in french",
        )
    ]
    sys = m._system_text(msgs, _params())
    assert "be terse" in sys and "answer in french" in sys


# --- unsupported-mode guards -----------------------------------------------


def test_rejects_tool_based_output_mode():
    m = ClaudeSDKModel("opus")
    with pytest.raises(UserError, match="PromptedOutput"):
        m._reject_unsupported(_params(output_mode="tool"))


def test_allows_prompted_and_text_modes():
    m = ClaudeSDKModel("opus")
    m._reject_unsupported(_params(output_mode="text"))
    m._reject_unsupported(_params(output_mode="prompted"))  # no raise


def test_rejects_function_tools():
    m = ClaudeSDKModel("opus")

    class _Tool:
        pass

    with pytest.raises(UserError, match="tool calling"):
        m._reject_unsupported(
            ModelRequestParameters(output_mode="text", function_tools=[_Tool()])
        )


# --- usage mapping ---------------------------------------------------------


def test_map_usage_from_result():
    class _R:
        usage: ClassVar = {
            "input_tokens": 10,
            "output_tokens": 5,
            "cache_read_input_tokens": 3,
            "cache_creation_input_tokens": 7,
        }

    u = _map_usage(_R())
    assert (u.input_tokens, u.output_tokens) == (10, 5)
    assert (u.cache_read_tokens, u.cache_write_tokens) == (3, 7)


def test_map_usage_handles_none_and_partial():
    assert _map_usage(None).input_tokens == 0

    class _R:
        usage: ClassVar = {"input_tokens": 4}

    assert _map_usage(_R()).output_tokens == 0


def test_map_usage_dict_full():
    u = map_usage_dict(
        {
            "input_tokens": 10,
            "output_tokens": 5,
            "cache_read_input_tokens": 3,
            "cache_creation_input_tokens": 7,
        }
    )
    assert (u.input_tokens, u.output_tokens) == (10, 5)
    # key renames: cache_read_input_tokens -> cache_read_tokens,
    # cache_creation_input_tokens -> cache_write_tokens
    assert (u.cache_read_tokens, u.cache_write_tokens) == (3, 7)


def test_map_usage_dict_partial_defaults_to_zero():
    u = map_usage_dict({"input_tokens": 4})
    assert u.input_tokens == 4
    assert (u.output_tokens, u.cache_read_tokens, u.cache_write_tokens) == (0, 0, 0)


def test_map_usage_dict_empty():
    u = map_usage_dict({})
    assert (u.input_tokens, u.output_tokens) == (0, 0)
    assert (u.cache_read_tokens, u.cache_write_tokens) == (0, 0)


def test_map_usage_dict_none():
    u = map_usage_dict(None)
    assert (u.input_tokens, u.output_tokens) == (0, 0)
    assert (u.cache_read_tokens, u.cache_write_tokens) == (0, 0)


def test_map_usage_dict_non_dict():
    for bad in (["input_tokens", 1], "input_tokens=1"):
        u = map_usage_dict(bad)
        assert (u.input_tokens, u.output_tokens) == (0, 0)
        assert (u.cache_read_tokens, u.cache_write_tokens) == (0, 0)


# --- per-model camelCase aggregation ---------------------------------------


def test_aggregate_per_model_single_model():
    from robotsix_llmio.claude_sdk._usage import _aggregate_per_model

    d = {
        "claude-3-5-haiku-20241022": {
            "inputTokens": 200,
            "outputTokens": 50,
            "cacheReadInputTokens": 10,
            "cacheCreationInputTokens": 0,
        }
    }
    out = _aggregate_per_model(d)
    assert out == {
        "input_tokens": 200,
        "output_tokens": 50,
        "cache_read_input_tokens": 10,
        "cache_creation_input_tokens": 0,
    }


def test_aggregate_per_model_multi_model():
    from robotsix_llmio.claude_sdk._usage import _aggregate_per_model

    d = {
        "claude-3-5-haiku-20241022": {"inputTokens": 200, "outputTokens": 50},
        "claude-3-5-sonnet-20241022": {"inputTokens": 100, "outputTokens": 30},
    }
    out = _aggregate_per_model(d)
    assert out == {"input_tokens": 300, "output_tokens": 80}


def test_aggregate_per_model_partial_keys():
    from robotsix_llmio.claude_sdk._usage import _aggregate_per_model

    d = {
        "claude-3-5-haiku-20241022": {"inputTokens": 200},
    }
    out = _aggregate_per_model(d)
    assert out == {"input_tokens": 200}


def test_aggregate_per_model_not_per_model_format():
    from robotsix_llmio.claude_sdk._usage import _aggregate_per_model

    # Values are not dicts → not the per-model format.
    assert _aggregate_per_model({"input_tokens": 4}) is None
    assert _aggregate_per_model({"a": 1, "b": 2}) is None


def test_aggregate_per_model_empty():
    from robotsix_llmio.claude_sdk._usage import _aggregate_per_model

    assert _aggregate_per_model({}) is None


def test_best_usage_dict_prefers_model_usage_per_model():
    """``_best_usage_dict`` aggregates the per-model camelCase
    ``model_usage`` and returns flat snake_case, ignoring ``usage``."""
    from robotsix_llmio.claude_sdk._usage import _best_usage_dict

    class _R:
        model_usage: ClassVar = {
            "claude-3-5-haiku-20241022": {"inputTokens": 200, "outputTokens": 50},
        }
        usage: ClassVar = {"input_tokens": 999, "output_tokens": 999}  # ignored

    out = _best_usage_dict(_R)
    assert out == {"input_tokens": 200, "output_tokens": 50}


def test_best_usage_dict_falls_back_to_usage():
    """When ``model_usage`` is absent, ``_best_usage_dict`` uses the flat
    ``usage`` dict."""
    from robotsix_llmio.claude_sdk._usage import _best_usage_dict

    class _R:
        usage: ClassVar = {"input_tokens": 10, "output_tokens": 5}

    out = _best_usage_dict(_R)
    assert out == {"input_tokens": 10, "output_tokens": 5}


def test_best_usage_dict_model_usage_empty_falls_through():
    """When ``model_usage`` is an empty dict, ``_best_usage_dict`` falls
    through to ``usage``."""
    from robotsix_llmio.claude_sdk._usage import _best_usage_dict

    class _R:
        model_usage: ClassVar = {}
        usage: ClassVar = {"input_tokens": 42, "output_tokens": 17}

    out = _best_usage_dict(_R)
    assert out == {"input_tokens": 42, "output_tokens": 17}


def test_map_usage_dict_from_per_model():
    """End-to-end: ``map_usage_dict`` receives the aggregated flat dict and
    returns correct ``RequestUsage``."""
    from robotsix_llmio.claude_sdk._usage import _aggregate_per_model, map_usage_dict

    d = {
        "claude-3-5-haiku-20241022": {
            "inputTokens": 200,
            "outputTokens": 50,
            "cacheReadInputTokens": 10,
            "cacheCreationInputTokens": 7,
        }
    }
    u = map_usage_dict(_aggregate_per_model(d))
    assert (u.input_tokens, u.output_tokens) == (200, 50)
    assert (u.cache_read_tokens, u.cache_write_tokens) == (10, 7)


# --- model identity --------------------------------------------------------


def test_model_name_defaults_to_sdk_model_and_system_is_anthropic():
    m = ClaudeSDKModel("haiku")
    assert m.model_name == "haiku"
    assert m.system == "anthropic"
    assert m.provider is None


# --- transient -------------------------------------------------------------


def test_sdk_subprocess_errors_are_transient():
    class CLIConnectionError(Exception):
        pass

    assert is_claude_sdk_transient(CLIConnectionError("lost cli")) is True


def test_plain_value_error_not_transient():
    assert is_claude_sdk_transient(ValueError("nope")) is False


def test_degenerate_success_is_transient_but_not_turn_limit():
    # The upstream SDK collapses a self-contradictory frame
    # (is_error=True, errors=[], subtype="success") into a bare
    # Exception("Claude Code returned an error result: success"). A re-run
    # clears it, so it must be retried locally — but it is NOT a turn-cap.
    e = Exception("Claude Code returned an error result: success")
    assert is_claude_sdk_transient(e) is True
    assert is_claude_sdk_turn_limit(e) is False


def test_genuine_error_subtypes_not_transient():
    # The match stays narrow: only the literal subtype="success" message is
    # transient. Genuine error subtypes must surface immediately.
    for subtype in ("error_during_execution", "error_max_turns"):
        e = Exception(f"Claude Code returned an error result: {subtype}")
        assert is_claude_sdk_transient(e) is False


def test_degenerate_success_matched_case_insensitively_through_chain():
    # Detected case-insensitively and through the cause/context chain.
    cause = RuntimeError("Claude Code RETURNED AN ERROR RESULT: SUCCESS")
    try:
        raise Exception("wrapper") from cause
    except Exception as e:
        assert is_claude_sdk_transient(e) is True


# --- turn-limit: hard failure, never retried -------------------------------


def test_turn_limit_message_detected_and_not_transient():
    e = Exception(
        "Claude Code returned an error result: Reached maximum number of turns (8)"
    )
    assert is_claude_sdk_turn_limit(e) is True
    # Must NOT be retried — retrying would just loop to the cap again.
    assert is_claude_sdk_transient(e) is False


def test_turn_limit_wins_even_when_wrapped_as_process_error():
    # ProcessError is normally transient; the turn-limit guard must win so we
    # fail loudly instead of burning retries.
    class ProcessError(Exception):
        pass

    e = ProcessError("CLI exited 1: Reached maximum number of turns (8)")
    assert is_claude_sdk_transient(e) is False


def test_turn_limit_error_type_detected_and_not_transient():
    from robotsix_llmio.claude_sdk.model import ClaudeSDKTurnLimitError

    e = ClaudeSDKTurnLimitError("hit the cap")
    assert is_claude_sdk_turn_limit(e) is True
    assert is_claude_sdk_transient(e) is False


def test_non_turn_limit_runtime_error_unaffected():
    assert is_claude_sdk_turn_limit(RuntimeError("something else")) is False


# --- per-call wall-clock timeout: stalled run fails fast + is retryable ------


def test_query_timeout_is_transient_but_not_turn_limit():
    from robotsix_llmio.claude_sdk.model import ClaudeSDKQueryTimeout

    e = ClaudeSDKQueryTimeout("stalled")
    # A stall re-runs cleanly, so it must be retried...
    assert is_claude_sdk_transient(e) is True
    # ...but it is NOT the (never-retried) turn-cap hard failure.
    assert is_claude_sdk_turn_limit(e) is False


def test_tool_loop_query_timeout_raises_claude_sdk_query_timeout(monkeypatch):
    """A query() that stalls past SDK_QUERY_TIMEOUT raises ClaudeSDKQueryTimeout
    (the tool-loop path), instead of hanging on the SDK's own backstop."""
    from robotsix_llmio.claude_sdk.model import ClaudeSDKQueryTimeout
    from robotsix_llmio.core import constants

    fake = _install_fake_sdk(monkeypatch)

    async def _hanging_query(*, prompt, options):
        await asyncio.sleep(30)  # never completes within the cap
        yield fake.ResultMessage()  # pragma: no cover — cancelled first

    fake.query = _hanging_query
    monkeypatch.setattr(constants, "SDK_QUERY_TIMEOUT", 0.05)

    provider = ClaudeSDKProvider()
    handle = provider.build_agent(
        level=1,
        tier_config=_HAIKU_AT_LEVEL1,
        system_prompt="sys",
        tools=[PydanticTool(_echo_sync, name="echo_sync")],
    )
    with pytest.raises(ClaudeSDKQueryTimeout):
        handle.run_sync("do something")
    handle.close()


def test_single_turn_invoke_query_timeout_raises(monkeypatch):
    """The no-tools single-turn path (ClaudeSDKModel._invoke) also enforces the
    per-call wall-clock cap."""
    from robotsix_llmio.claude_sdk.model import ClaudeSDKModel, ClaudeSDKQueryTimeout
    from robotsix_llmio.core import constants

    fake = _install_fake_sdk(monkeypatch)

    async def _hanging_query(*, prompt, options):
        await asyncio.sleep(30)
        yield fake.ResultMessage()  # pragma: no cover — cancelled first

    fake.query = _hanging_query
    monkeypatch.setattr(constants, "SDK_QUERY_TIMEOUT", 0.05)

    model = ClaudeSDKModel("haiku")
    with pytest.raises(ClaudeSDKQueryTimeout):
        asyncio.run(model._invoke("hi", None))


# --- turn cap: single source, generous for injected-MCP-tool loops ---------


def test_tool_handle_uses_shared_max_turns_cap():
    from robotsix_llmio.claude_sdk.model import _MAX_TURNS

    handle = _SdkToolAgentHandle("opus", "sys", None, [], str)
    assert handle._max_turns == _MAX_TURNS  # single source — paths can't drift
    assert _MAX_TURNS >= 100  # generous cap so genuine tool loops don't trip it


# ---------------------------------------------------------------------------
# Helpers for tool-loop bridge tests
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
# Tool-loop bridge tests
# ---------------------------------------------------------------------------


def _echo_sync(text: str) -> str:
    """Echo the input."""
    return text


def test_build_agent_model_override_tool_path(monkeypatch):
    """When ``model`` is provided on the tool path, ``_SdkToolAgentHandle``
    receives the explicit model name, bypassing tier_config."""
    fake = _install_fake_sdk(monkeypatch)

    canned_text = "explicit model override works"

    async def _fake_query(*, prompt, options):
        yield fake.AssistantMessage(canned_text)
        yield fake.ResultMessage({"input_tokens": 1, "output_tokens": 1})

    fake.query = _fake_query

    provider = ClaudeSDKProvider()
    handle = provider.build_agent(
        level=1,
        tier_config=_HAIKU_AT_LEVEL1,
        model="sonnet",  # explicit override — not "haiku" from tier_config
        system_prompt="sys",
        tools=[PydanticTool(_echo_sync, name="echo_sync")],
    )

    assert isinstance(handle, _SdkToolAgentHandle)
    assert handle._sdk_model == "sonnet"  # type: ignore[attr-defined]

    result = handle.run_sync("use the tool")
    assert result.output == canned_text
    handle.close()


def test_tool_path_resolves_bare_model_name(monkeypatch):
    """The tool path must pass the SDK the *bare* model name ("haiku"), not the
    transport-prefixed id ("claudeSDK-haiku") — the prefixed id is unknown to the
    SDK and yields a degenerate "error result" frame."""
    _install_fake_sdk(monkeypatch)
    provider = ClaudeSDKProvider()
    handle = provider.build_agent(
        level=1,
        tier_config=_HAIKU_AT_LEVEL1,  # model="claudeSDK-haiku", model_name="haiku"
        system_prompt="sys",
        tools=[PydanticTool(_echo_sync, name="echo_sync")],
    )
    assert isinstance(handle, _SdkToolAgentHandle)
    assert handle._sdk_model == "haiku"  # type: ignore[attr-defined]
    handle.close()


def test_restricted_tool_path_denies_builtins_not_mcp(monkeypatch):
    """``builtin_tools=False`` denies the built-in tools by explicit name (not a
    ``"*"`` wildcard, which would also block the injected MCP tools) and sets no
    allow-list; ``builtin_tools=True`` keeps full access (allow-list, no deny)."""
    from robotsix_llmio.claude_sdk._tool_agent import _BUILTIN_TOOL_DENYLIST

    _install_fake_sdk(monkeypatch)
    provider = ClaudeSDKProvider()

    restricted = provider.build_agent(
        level=1,
        tier_config=_HAIKU_AT_LEVEL1,
        system_prompt="sys",
        tools=[PydanticTool(_echo_sync, name="echo_sync")],
        builtin_tools=False,
    )
    opts = restricted._build_options("sys")  # type: ignore[attr-defined]
    assert opts.disallowed_tools == list(_BUILTIN_TOOL_DENYLIST)
    assert "Bash" in opts.disallowed_tools
    assert "*" not in opts.disallowed_tools
    assert opts.permission_mode == "bypassPermissions"
    assert not hasattr(opts, "allowed_tools")
    restricted.close()

    full = provider.build_agent(
        level=1,
        tier_config=_HAIKU_AT_LEVEL1,
        system_prompt="sys",
        tools=[PydanticTool(_echo_sync, name="echo_sync")],
        builtin_tools=True,
    )
    opts2 = full._build_options("sys")  # type: ignore[attr-defined]
    assert not hasattr(opts2, "disallowed_tools")
    assert hasattr(opts2, "allowed_tools")
    full.close()


def test_tool_agent_invokes_tool_and_returns_output(monkeypatch):
    """build_agent with tools returns a handle; run_sync invokes the SDK tool
    loop and the final text reaches .output (offline, monkeypatched SDK)."""
    fake = _install_fake_sdk(monkeypatch)

    canned_text = "the echo tool says: hello world"

    async def _fake_query(*, prompt, options):
        yield fake.AssistantMessage(canned_text)
        yield fake.ResultMessage({"input_tokens": 10, "output_tokens": 5})

    fake.query = _fake_query

    provider = ClaudeSDKProvider()
    handle = provider.build_agent(
        level=1,
        tier_config=_HAIKU_AT_LEVEL1,
        system_prompt="You are a tester.",
        tools=[PydanticTool(_echo_sync, name="echo_sync")],
    )

    assert isinstance(handle, _SdkToolAgentHandle)

    result = handle.run_sync("Use the echo tool")
    assert isinstance(result, _SdkToolResult)
    assert result.output == canned_text
    assert isinstance(result.all_messages(), list)
    assert len(result.all_messages()) == 1
    assert result.usage.input_tokens == 10
    assert result.usage.output_tokens == 5

    # Tool was registered with correct metadata
    assert len(fake._tool_regs) == 1
    assert fake._tool_regs[0]["name"] == "echo_sync"
    assert "Echo the input" in (fake._tool_regs[0]["description"] or "")
    assert fake._tool_regs[0]["schema"]["type"] == "object"

    # MCP server created with the tool
    assert len(fake._server_calls) == 1
    assert fake._server_calls[0]["name"] == "milltools"
    assert len(fake._server_calls[0]["tools"]) == 1

    handle.close()  # no-op, must not raise


def test_tool_definition_mapping_from_pydantic_tool(monkeypatch):
    """SDK tool registration receives correct name/description/schema from a
    pydantic-ai ``Tool`` with explicit metadata."""
    fake = _install_fake_sdk(monkeypatch)

    def _add(a: int, b: int = 0) -> int:
        """Add two numbers."""
        return a + b

    tool = PydanticTool(_add, name="adder", description="Returns a + b.")
    _convert_tools([tool])

    assert len(fake._tool_regs) == 1
    reg = fake._tool_regs[0]
    assert reg["name"] == "adder"
    assert reg["description"] == "Returns a + b."
    assert reg["schema"]["type"] == "object"
    assert "a" in reg["schema"]["properties"]
    assert "b" in reg["schema"]["properties"]
    assert reg["schema"]["properties"]["a"]["type"] == "integer"


def test_tool_definition_mapping_from_plain_callable(monkeypatch):
    """Plain callable is normalised to ``Tool`` and SDK registration still
    receives correct metadata derived from docstring + type hints."""
    fake = _install_fake_sdk(monkeypatch)

    def greet(name: str, enthusiastic: bool = False) -> str:
        """Return a greeting for *name*.

        If *enthusiastic*, uppercase the result.
        """
        msg = f"Hello {name}"
        return msg.upper() if enthusiastic else msg

    _convert_tools([greet])

    assert len(fake._tool_regs) == 1
    reg = fake._tool_regs[0]
    assert reg["name"] == "greet"
    assert "Return a greeting" in (reg["description"] or "")
    assert reg["schema"]["type"] == "object"
    assert "name" in reg["schema"]["properties"]
    assert reg["schema"]["properties"]["name"]["type"] == "string"


def test_takes_ctx_tool_raises_user_error_at_build_time(monkeypatch):
    """A tool with ``takes_ctx=True`` raises ``UserError`` at conversion time,
    not at invocation time — matching ``_reject_unsupported`` fail-fast."""
    _install_fake_sdk(monkeypatch)

    def _ctx_tool(ctx: RunContext, x: int) -> str:
        return str(x)

    ctx_tool = PydanticTool(_ctx_tool, takes_ctx=True)

    with pytest.raises(UserError, match=r"takes_ctx|RunContext"):
        _convert_tools([ctx_tool])


# ---------------------------------------------------------------------------
# run_sync/run kwargs: honor message_history, warn on the rest (never silent)
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


def test_tool_run_sync_honors_message_history(monkeypatch):
    """A message_history passed to the tool-loop run_sync is folded into the
    prompt (prior transcript + the new turn), so the caller keeps context."""
    fake = _install_fake_sdk(monkeypatch)
    captured: dict = {}
    fake.query = _capturing_query(fake, captured)

    handle = _tool_handle()
    history = [
        ModelRequest(parts=[UserPromptPart(content="first question")]),
        ModelResponse(parts=[TextPart(content="prior answer")]),
    ]
    handle.run_sync("the new turn", message_history=history)

    prompt = captured["prompt"]
    assert "first question" in prompt
    assert "prior answer" in prompt
    assert prompt.endswith("User: the new turn")  # new turn appended last
    handle.close()


def test_tool_run_sync_without_history_sends_prompt_verbatim(monkeypatch):
    fake = _install_fake_sdk(monkeypatch)
    captured: dict = {}
    fake.query = _capturing_query(fake, captured)

    handle = _tool_handle()
    handle.run_sync("just this")
    assert captured["prompt"] == "just this"  # no history → no transcript wrap
    handle.close()


def test_tool_run_sync_warns_on_unsupported_kwargs(monkeypatch, caplog):
    """Unsupported run kwargs (usage_limits, model_settings) are warned about,
    not silently dropped — and the run still completes."""
    fake = _install_fake_sdk(monkeypatch)
    captured: dict = {}
    fake.query = _capturing_query(fake, captured)

    handle = _tool_handle()
    with caplog.at_level(logging.WARNING, logger="robotsix_llmio.claude_sdk"):
        result = handle.run_sync(
            "hi", usage_limits="L", model_settings={"temperature": 0}
        )

    assert result.output == "done"  # run still works
    warned = " ".join(r.getMessage() for r in caplog.records)
    assert "usage_limits" in warned
    assert "model_settings" in warned
    handle.close()


def test_tool_async_run_honors_message_history(monkeypatch):
    """The async run() path threads message_history through the same way."""
    fake = _install_fake_sdk(monkeypatch)
    captured: dict = {}
    fake.query = _capturing_query(fake, captured)

    handle = _tool_handle()
    history = [ModelRequest(parts=[UserPromptPart(content="earlier ctx")])]
    asyncio.run(handle.run("now", message_history=history))

    assert "earlier ctx" in captured["prompt"]
    assert captured["prompt"].endswith("User: now")
    handle.close()


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


class _FakeThinkingBlock:
    """A fake SDK thinking block — matched by class name in ``_stream_query``."""

    def __init__(self, thinking: str) -> None:
        self.thinking = thinking


_FakeThinkingBlock.__name__ = "ThinkingBlock"


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


# ---------------------------------------------------------------------------
# Live tool round-trip (deselected by default)
# ---------------------------------------------------------------------------


@pytest.mark.live
def test_live_tool_round_trip():
    """Live: one real tool call end-to-end with the Claude Agent SDK.

    Skips when the ``claude`` CLI / SDK is unavailable or not logged in.
    """
    import shutil

    if shutil.which("claude") is None:
        pytest.skip("claude CLI not found on PATH")

    try:
        pass
    except Exception:
        pytest.skip("claude_agent_sdk import failed (SDK not installed)")

    provider = ClaudeSDKProvider()

    def _echo(text: str) -> str:
        return text

    handle = provider.build_agent(
        level=1,
        tier_config=_HAIKU_AT_LEVEL1,
        system_prompt="You are a QA bot. When asked to echo, call the echo "
        'tool and then repeat exactly what it returned prefixed with "ECHO: ".',
        tools=[PydanticTool(_echo)],
    )

    result = handle.run_sync("Use the echo tool to repeat: hello42")
    assert "hello42" in str(result.output).lower()
    handle.close()


@pytest.mark.live
def test_live_query_timeout_fires_against_real_cli(monkeypatch):
    """Live: a real ``query()`` against the ``claude`` CLI subprocess that is
    capped at a sub-spawn-time wall clock raises ``ClaudeSDKQueryTimeout`` (not
    a hang) — proving the asyncio.wait_for cancellation path works end-to-end
    against the real subprocess, not just the offline fake."""
    import shutil

    if shutil.which("claude") is None:
        pytest.skip("claude CLI not found on PATH")

    from robotsix_llmio.claude_sdk.model import ClaudeSDKQueryTimeout
    from robotsix_llmio.core import constants

    # 1ms cap — far below the time to even spawn the CLI, so it must trip.
    monkeypatch.setattr(constants, "SDK_QUERY_TIMEOUT", 0.001)

    provider = ClaudeSDKProvider()

    def _echo(text: str) -> str:
        return text

    handle = provider.build_agent(
        level=1,
        tier_config=_HAIKU_AT_LEVEL1,
        system_prompt="You are a QA bot.",
        tools=[PydanticTool(_echo)],
    )
    with pytest.raises(ClaudeSDKQueryTimeout):
        handle.run_sync("Use the echo tool to repeat: hello42")
    handle.close()


@pytest.mark.live
def test_live_tool_run_sync_honors_message_history():
    """Live: a real tool-loop run_sync recalls context supplied only via
    message_history — proving the folded-in transcript actually reaches the
    model, not just the offline fake."""
    import shutil

    if shutil.which("claude") is None:
        pytest.skip("claude CLI not found on PATH")

    def _noop(text: str) -> str:
        """A trivial tool so this exercises the tool-loop path."""
        return text

    handle = ClaudeSDKProvider().build_agent(
        level=1,
        tier_config=_HAIKU_AT_LEVEL1,
        system_prompt="You are a precise assistant. Answer tersely.",
        tools=[PydanticTool(_noop, name="noop")],
    )

    # The fact lives ONLY in the prior turn passed as message_history.
    history = [
        ModelRequest(
            parts=[UserPromptPart(content="My favorite number is 4273. Acknowledge.")]
        ),
        ModelResponse(parts=[TextPart(content="Acknowledged: 4273.")]),
    ]
    result = handle.run_sync(
        "What is my favorite number? Reply with just the digits.",
        message_history=history,
    )
    assert "4273" in str(result.output)
    handle.close()


# --- structured-output JSON extraction (prose + fenced / stray braces) -------


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


def test_parse_output_str_passthrough():
    assert _parse_output("anything", str) == "anything"


def test_extract_clean_json():
    assert _extract_json_object('{"verdict": "APPROVE"}') == {"verdict": "APPROVE"}


def test_extract_fenced_json_after_prose():
    # The 402b shape: prose preamble, then a ```json fence with the verdict.
    text = (
        "Looking at this review.\n\n## Analysis\nlooks good.\n\n"
        '```json\n{"verdict": "APPROVE", "auto_merge_eligible": true}\n```\n'
    )
    v = _parse_output(text, _Verdict)
    assert isinstance(v, _Verdict)
    assert v.verdict == "APPROVE" and v.auto_merge_eligible is True


def test_extract_ignores_stray_prose_brace():
    # A stray `{...}` in prose must NOT derail extraction of the real object
    # (the old greedy re.search anchored on the first brace and failed).
    text = (
        "The `{verified_proposals}` kwarg is passed through. Verdict below:\n"
        '```json\n{"verdict": "REQUEST_CHANGES"}\n```'
    )
    v = _parse_output(text, _Verdict)
    assert v.verdict == "REQUEST_CHANGES"


def test_extract_prose_wrapped_json_no_fence():
    # No fence, just prose then a JSON object with nested structures.
    text = (
        'Here is my verdict: {"verdict": "APPROVE", "auto_merge_eligible": false} done.'
    )
    v = _parse_output(text, _Verdict)
    assert v.verdict == "APPROVE"


def test_extract_picks_last_valid_object():
    # An earlier non-matching object (e.g. an example) then the real one.
    text = (
        'Example shape: {"foo": 1}\n\nActual:\n'
        '```json\n{"verdict": "NEEDS_DISCUSSION"}\n```'
    )
    v = _parse_output(text, _Verdict)
    assert v.verdict == "NEEDS_DISCUSSION"


def test_extract_no_json_falls_back_to_text():
    import pytest

    with pytest.raises(ValueError, match="no JSON object found"):
        _parse_output("no json at all here", _Verdict)


def test_extract_nested_object_captured_whole():
    text = '```json\n{"verdict": "APPROVE", "nested": {"a": {"b": [1,2]}}}\n```'
    assert _extract_json_object(text) == {
        "verdict": "APPROVE",
        "nested": {"a": {"b": [1, 2]}},
    }


# --- multi-output structured output (PromptedOutput) -----------------------


def test_parse_output_multi_output_first_type_matches():
    """A-shaped JSON validates against the first model in PromptedOutput."""
    from pydantic_ai import PromptedOutput

    text = '{"verdict": "APPROVE", "auto_merge_eligible": false}'
    result = _parse_output(text, PromptedOutput([_Verdict, _AltVerdict]))
    assert isinstance(result, _Verdict)
    assert result.verdict == "APPROVE"


def test_parse_output_multi_output_second_type_matches():
    """B-shaped JSON falls through to the second model in PromptedOutput."""
    from pydantic_ai import PromptedOutput

    text = '{"outcome": "rejected"}'
    result = _parse_output(text, PromptedOutput([_Verdict, _AltVerdict]))
    assert isinstance(result, _AltVerdict)
    assert result.outcome == "rejected"


def test_parse_output_multi_output_no_match_raises():
    """JSON matching no declared model raises (not silently returns str)."""
    import pytest
    from pydantic import ValidationError
    from pydantic_ai import PromptedOutput

    text = '{"unrecognised_field": 42}'
    with pytest.raises(ValidationError):
        _parse_output(text, PromptedOutput([_Verdict, _AltVerdict]))


def test_prepare_prompt_multi_output_anyof_schema():
    """System prompt contains anyOf when PromptedOutput wraps multiple types."""
    from pydantic_ai import PromptedOutput

    handle = _make_minimal_handle(output_type=PromptedOutput([_Verdict, _AltVerdict]))
    _, system_prompt = handle._prepare_prompt("hello", None)
    assert "anyOf" in system_prompt


def test_prepare_prompt_single_prompted_output_no_anyof():
    """Single-model PromptedOutput uses a flat schema (no anyOf)."""
    from pydantic_ai import PromptedOutput

    handle = _make_minimal_handle(output_type=PromptedOutput(_Verdict))
    _, system_prompt = handle._prepare_prompt("hello", None)
    assert "anyOf" not in system_prompt
    assert "verdict" in system_prompt


def test_sdk_tool_agent_handle_rejects_list_output_type():
    """Bare list output_type raises UserError at construction time."""
    import pytest
    from pydantic_ai.exceptions import UserError

    with pytest.raises(UserError, match="list/union output_type is not supported"):
        _make_minimal_handle(output_type=[_Verdict, _AltVerdict])


# ---------------------------------------------------------------------------
# build_agent output_type wrapping for claude-sdk no-tools path
# ---------------------------------------------------------------------------


def test_build_agent_level1_raw_model_wrapped_in_prompted_output():
    """At level=1 with a raw pydantic model, the no-tools path wraps it in
    PromptedOutput before delegating to super(), so ClaudeSDKModel does not
    reject it."""
    provider = ClaudeSDKProvider()
    handle = provider.build_agent(
        level=1,
        tier_config=_HAIKU_AT_LEVEL1,
        system_prompt="You are helpful.",
        output_type=_Verdict,
        tools=None,
    )
    from pydantic_ai import PromptedOutput

    assert isinstance(handle._agent.output_type, PromptedOutput)  # type: ignore[attr-defined]
    unwrapped = handle._agent.output_type.outputs  # type: ignore[attr-defined]
    assert unwrapped is _Verdict
    handle.close()


def test_build_agent_level1_str_output_type_unchanged():
    """str output_type at level=1 is not wrapped (already a valid type)."""
    provider = ClaudeSDKProvider()
    handle = provider.build_agent(
        level=1,
        tier_config=_HAIKU_AT_LEVEL1,
        system_prompt="You are helpful.",
        output_type=str,
        tools=[],
    )
    assert handle._agent.output_type is str  # type: ignore[attr-defined]
    handle.close()


def test_build_agent_level1_already_wrapped_no_double_wrap():
    """When the caller passes PromptedOutput explicitly at level=1, it is not
    double-wrapped."""
    from pydantic_ai import PromptedOutput

    provider = ClaudeSDKProvider()
    handle = provider.build_agent(
        level=1,
        tier_config=_HAIKU_AT_LEVEL1,
        system_prompt="You are helpful.",
        output_type=PromptedOutput(_Verdict),
        tools=None,
    )
    # Should still be a single PromptedOutput wrapping _Verdict.
    assert isinstance(handle._agent.output_type, PromptedOutput)  # type: ignore[attr-defined]
    assert handle._agent.output_type.outputs is _Verdict  # type: ignore[attr-defined]
    handle.close()


def test_build_agent_level2_raw_model_still_works():
    """At level=2 the local wrap applies (output_type is not yet a marker),
    then _resolve_output_type sees PromptedOutput and passes it through —
    no double-wrap, and the agent builds correctly."""
    provider = ClaudeSDKProvider()
    handle = provider.build_agent(
        level=2,
        tier_config=_OPUS_AT_LEVEL2,
        system_prompt="You are helpful.",
        output_type=_Verdict,
        tools=None,
    )
    from pydantic_ai import PromptedOutput

    assert isinstance(handle._agent.output_type, PromptedOutput)  # type: ignore[attr-defined]
    assert handle._agent.output_type.outputs is _Verdict  # type: ignore[attr-defined]
    handle.close()


def test_build_agent_tool_path_output_type_unaffected(monkeypatch):
    """The tool path passes output_type through to _SdkToolAgentHandle
    unchanged — the local wrap only runs on the no-tools branch."""
    fake = _install_fake_sdk(monkeypatch)

    async def _fake_query(*, prompt, options):
        yield fake.AssistantMessage("done")
        yield fake.ResultMessage({"input_tokens": 1, "output_tokens": 1})

    fake.query = _fake_query

    provider = ClaudeSDKProvider()
    handle = provider.build_agent(
        level=1,
        tier_config=_HAIKU_AT_LEVEL1,
        system_prompt="sys",
        tools=[PydanticTool(_echo_sync, name="echo_sync")],
        output_type=_Verdict,
    )
    # The tool path stores output_type directly — no PromptedOutput wrap.
    assert handle._output_type is _Verdict  # type: ignore[attr-defined]
    handle.close()

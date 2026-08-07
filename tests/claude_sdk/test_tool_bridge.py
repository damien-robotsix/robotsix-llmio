"""Tool-loop bridge tests — tool conversion, SDK registration,
builtin-tool control, and ctx-handling edge cases.  Offline only
(monkeypatched ``claude_agent_sdk``)."""

from __future__ import annotations

import pytest

pytest.importorskip("pydantic_ai")

from pydantic_ai.exceptions import UserError
from pydantic_ai.tools import RunContext
from pydantic_ai.tools import Tool as PydanticTool

from robotsix_llmio.claude_sdk._tool_agent import (
    _BUILTIN_TOOL_DENYLIST,
    _WEB_TOOL_NAMES,
    _SdkToolAgentHandle,
    _SdkToolResult,
)
from robotsix_llmio.claude_sdk._tool_converter import _convert_tools
from robotsix_llmio.claude_sdk.provider import (
    ClaudeSDKProvider,
)

from .conftest import _HAIKU_AT_LEVEL1, _echo_sync, _install_fake_sdk

# ---------------------------------------------------------------------------
# Tool-loop bridge tests
# ---------------------------------------------------------------------------


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
    assert opts.disallowed_tools == list(_BUILTIN_TOOL_DENYLIST) + _WEB_TOOL_NAMES
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


def test_required_ctx_tool_raises_user_error_at_build_time(monkeypatch):
    """A tool whose ctx parameter has NO default raises ``UserError`` at
    conversion time — the Claude SDK cannot supply a RunContext."""
    _install_fake_sdk(monkeypatch)

    def _ctx_tool(ctx: RunContext, x: int) -> str:
        return str(x)

    ctx_tool = PydanticTool(_ctx_tool, takes_ctx=True)

    with pytest.raises(UserError, match=r"required.*RunContext|no default"):
        _convert_tools([ctx_tool])


def test_optional_ctx_tool_converts_successfully(monkeypatch):
    """A tool whose ctx parameter has a default (e.g. None) converts without
    raising — the function is called without ctx, letting the default apply."""
    fake = _install_fake_sdk(monkeypatch)

    def _opt_ctx_tool(
        ctx: RunContext[None] = None, *, path: str, offset: int = 1
    ) -> str:
        """Read a file at *path* starting from *offset*."""
        return f"{path}:{offset}"

    ctx_tool = PydanticTool(_opt_ctx_tool, takes_ctx=True)

    _convert_tools([ctx_tool])

    assert len(fake._tool_regs) == 1
    reg = fake._tool_regs[0]
    assert reg["name"] == "_opt_ctx_tool"
    assert "Read a file" in (reg["description"] or "")
    assert reg["schema"]["type"] == "object"
    # The ctx parameter must not appear in the tool schema.
    assert "ctx" not in reg["schema"]["properties"]
    assert "path" in reg["schema"]["properties"]
    assert "offset" in reg["schema"]["properties"]


def test_required_ctx_no_default_still_raises(monkeypatch):
    """A tool with ctx parameter lacking a default still raises the descriptive
    UserError — only optional-ctx tools are accepted."""
    _install_fake_sdk(monkeypatch)

    def _required_ctx(ctx: RunContext[int], x: int) -> str:
        return str(ctx.deps + x)

    tool = PydanticTool(_required_ctx, takes_ctx=True)

    with pytest.raises(UserError, match=r"required.*RunContext|no default"):
        _convert_tools([tool])


def test_web_tools_opt_in_keeps_the_rest_denied(monkeypatch) -> None:
    """``web_tools=True`` grants WebFetch/WebSearch without reopening the
    sandbox the denylist exists to enforce.

    A restricted research agent asked to check three CVE advisories reported
    "11 sources fetched, all empty" — the calls had been refused, and a refusal
    is indistinguishable from an empty result to the model. Reading the web
    mutates nothing local, so it is separable from the filesystem and shell
    restrictions.
    """
    _install_fake_sdk(monkeypatch)
    provider = ClaudeSDKProvider()

    agent = provider.build_agent(
        level=1,
        tier_config=_HAIKU_AT_LEVEL1,
        system_prompt="sys",
        tools=[PydanticTool(_echo_sync, name="echo_sync")],
        builtin_tools=False,
        web_tools=True,
    )
    opts = agent._build_options("sys")  # type: ignore[attr-defined]
    for name in _WEB_TOOL_NAMES:
        assert name not in opts.disallowed_tools
    # Everything the sandbox actually protects stays denied.
    for name in ("Bash", "Read", "Write", "Edit", "Glob"):
        assert name in opts.disallowed_tools
    agent.close()


def test_web_tools_default_off(monkeypatch) -> None:
    """An agent that never needs the web should not carry the capability."""
    _install_fake_sdk(monkeypatch)
    provider = ClaudeSDKProvider()

    agent = provider.build_agent(
        level=1,
        tier_config=_HAIKU_AT_LEVEL1,
        system_prompt="sys",
        tools=[PydanticTool(_echo_sync, name="echo_sync")],
        builtin_tools=False,
    )
    opts = agent._build_options("sys")  # type: ignore[attr-defined]
    for name in _WEB_TOOL_NAMES:
        assert name in opts.disallowed_tools
    agent.close()

"""Live tool round-trip tests (deselected by default).

These tests require the ``claude`` CLI / SDK installed and logged in.
Run with:  pytest -m live
"""

from __future__ import annotations

import pytest

pytest.importorskip("pydantic_ai")

from pydantic_ai.messages import (
    ModelRequest,
    ModelResponse,
    TextPart,
    UserPromptPart,
)
from pydantic_ai.tools import Tool as PydanticTool

from robotsix_llmio.claude_sdk.provider import ClaudeSDKProvider

from .conftest import _HAIKU_AT_LEVEL1

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

    from robotsix_llmio.claude_sdk._errors import ClaudeSDKQueryTimeout
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

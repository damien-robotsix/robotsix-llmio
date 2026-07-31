"""Transient-error classification, turn-limit detection, usage-exhaustion
guards, per-call timeout, turn cap, unsupported-mode guards, and model
identity — extracted from ``test_claude_sdk.py`` so the transient logic
can be tested in isolation."""

from __future__ import annotations

import asyncio

import pytest

pytest.importorskip("pydantic_ai")

from pydantic_ai.exceptions import UserError
from pydantic_ai.models import ModelRequestParameters
from pydantic_ai.tools import Tool as PydanticTool

from robotsix_llmio.claude_sdk._model import ClaudeSDKModel
from robotsix_llmio.claude_sdk._tool_agent import _SdkToolAgentHandle
from robotsix_llmio.claude_sdk.provider import ClaudeSDKProvider
from robotsix_llmio.claude_sdk.transient import (
    is_claude_sdk_transient,
    is_claude_sdk_turn_limit,
)

from .conftest import _HAIKU_AT_LEVEL1, _echo_sync, _install_fake_sdk

# ---------------------------------------------------------------------------
# helper (extracted together with the unsupported-mode guard tests)
# ---------------------------------------------------------------------------


def _params(output_mode="text"):
    return ModelRequestParameters(output_mode=output_mode)


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
    from robotsix_llmio.claude_sdk._errors import ClaudeSDKTurnLimitError

    e = ClaudeSDKTurnLimitError("hit the cap")
    assert is_claude_sdk_turn_limit(e) is True
    assert is_claude_sdk_transient(e) is False


def test_non_turn_limit_runtime_error_unaffected():
    assert is_claude_sdk_turn_limit(RuntimeError("something else")) is False


# --- usage exhaustion: hard failure, never retried at the same tier --------


def test_usage_exhausted_error_type_detected_and_not_transient():
    from robotsix_llmio.claude_sdk._errors import ClaudeSDKUsageExhaustedError
    from robotsix_llmio.claude_sdk.transient import is_claude_sdk_usage_exhausted

    e = ClaudeSDKUsageExhaustedError("You're out of usage credits")
    assert is_claude_sdk_usage_exhausted(e) is True
    # Must NOT be retried at the same tier — the credits stay exhausted.
    assert is_claude_sdk_transient(e) is False


def test_usage_exhausted_detected_through_chain():
    from robotsix_llmio.claude_sdk._errors import ClaudeSDKUsageExhaustedError
    from robotsix_llmio.claude_sdk.transient import is_claude_sdk_usage_exhausted

    cause = ClaudeSDKUsageExhaustedError("out of usage credits")
    try:
        raise Exception("wrapper") from cause
    except Exception as e:
        assert is_claude_sdk_usage_exhausted(e) is True
        assert is_claude_sdk_transient(e) is False


def test_non_usage_exhausted_runtime_error_unaffected():
    from robotsix_llmio.claude_sdk.transient import is_claude_sdk_usage_exhausted

    assert is_claude_sdk_usage_exhausted(RuntimeError("something else")) is False


def test_is_usage_exhausted_text_matches_case_insensitively():
    from robotsix_llmio.claude_sdk.transient import is_usage_exhausted_text

    assert is_usage_exhausted_text("You're OUT OF USAGE CREDITS · resets soon")
    assert is_usage_exhausted_text("out of usage credits") is True
    assert is_usage_exhausted_text("all good here") is False


# --- per-call wall-clock timeout: stalled run fails fast + is retryable ------


def test_query_timeout_is_transient_but_not_turn_limit():
    from robotsix_llmio.claude_sdk._errors import ClaudeSDKQueryTimeout

    e = ClaudeSDKQueryTimeout("stalled")
    # A stall re-runs cleanly, so it must be retried...
    assert is_claude_sdk_transient(e) is True
    # ...but it is NOT the (never-retried) turn-cap hard failure.
    assert is_claude_sdk_turn_limit(e) is False


def test_tool_loop_query_timeout_raises_claude_sdk_query_timeout(monkeypatch):
    """A query() that stalls past SDK_QUERY_TIMEOUT raises ClaudeSDKQueryTimeout
    (the tool-loop path), instead of hanging on the SDK's own backstop."""
    from robotsix_llmio.claude_sdk._errors import ClaudeSDKQueryTimeout
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
    from robotsix_llmio.claude_sdk._errors import ClaudeSDKQueryTimeout
    from robotsix_llmio.claude_sdk._model import ClaudeSDKModel
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
    from robotsix_llmio.claude_sdk._model import _MAX_TURNS

    handle = _SdkToolAgentHandle("opus", "sys", None, [], str)
    assert handle._max_turns == _MAX_TURNS  # single source — paths can't drift
    assert _MAX_TURNS >= 100  # generous cap so genuine tool loops don't trip it

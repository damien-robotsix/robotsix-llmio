"""Transient-error classification, turn-limit detection, usage-exhaustion
guards, per-call timeout, turn cap, unsupported-mode guards, and model
identity — extracted from ``test_claude_sdk.py`` so the transient logic
can be tested in isolation."""

from __future__ import annotations

import asyncio
import errno

import pytest

pytest.importorskip("pydantic_ai")

from pydantic_ai.exceptions import UserError
from pydantic_ai.models import ModelRequestParameters
from pydantic_ai.tools import Tool as PydanticTool

from robotsix_llmio.claude_sdk._model import ClaudeSDKModel
from robotsix_llmio.claude_sdk._tool_agent import _SdkToolAgentHandle
from robotsix_llmio.claude_sdk.provider import ClaudeSDKProvider
from robotsix_llmio.claude_sdk.transient import (
    is_claude_sdk_session_option_error,
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
    # The three cap wordings the CLI has been seen to use (2026-08-27/29).
    assert is_usage_exhausted_text(
        "You've hit your session limit · resets 11:10am (UTC)"
    )
    assert is_usage_exhausted_text("You've hit your limit · resets 8pm (UTC)")
    assert is_usage_exhausted_text("You've hit your weekly limit · resets Monday")
    assert is_usage_exhausted_text("all good here") is False


def test_session_limit_text_is_usage_exhausted():
    """The rolling session window must be fallback-eligible, not transient.

    Observed on robotsix-chat 2026-08-27. Matching no signature, this text
    fell through to the degenerate-success frame the CLI wraps it in, was
    classified transient, burned all three retries against the same limited
    tier, and reached the user as an opaque internal error — while the
    fallback model, gated on ClaudeSDKUsageExhaustedError, was never tried.
    """
    from robotsix_llmio.claude_sdk.transient import is_usage_exhausted_text

    assert is_usage_exhausted_text(
        "You've hit your session limit \u00b7 resets 11:10am (UTC)"
    )
    assert is_usage_exhausted_text("Claude usage limit reached") is True


def test_session_limit_is_not_transient_even_as_a_degenerate_success():
    """The degenerate-success wrapper must not win over the session limit.

    ``is_claude_sdk_transient`` checks the terminal classes first precisely so
    a wrapped permanent cause is not retried; this pins that ordering for the
    session-limit case.
    """
    from robotsix_llmio.claude_sdk._errors import ClaudeSDKUsageExhaustedError
    from robotsix_llmio.claude_sdk.transient import is_claude_sdk_transient

    exc = ClaudeSDKUsageExhaustedError(
        "You've hit your session limit \u00b7 resets 11:10am (UTC)"
    )
    exc.__cause__ = RuntimeError("Claude Code returned an error result: success")

    assert is_claude_sdk_transient(exc) is False


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


# --- API 400: request validation, never retryable --------------------------


def test_permanent_api_error_type_detected_and_not_transient():
    from robotsix_llmio.claude_sdk._errors import ClaudeSDKPermanentAPIError
    from robotsix_llmio.claude_sdk.transient import is_claude_sdk_permanent_api_error

    e = ClaudeSDKPermanentAPIError("API Error: 400 bad param")
    assert is_claude_sdk_permanent_api_error(e) is True
    # Re-sending the identical invalid request reproduces it exactly.
    assert is_claude_sdk_transient(e) is False


def test_permanent_api_error_beats_degenerate_success_wrapper():
    """The regression that let a 400 burn all 3 retries: the SDK collapses the
    error frame into its degenerate-success message, which IS transient. The
    permanent cause must win when both signatures are in the chain."""
    from robotsix_llmio.claude_sdk._errors import ClaudeSDKPermanentAPIError
    from robotsix_llmio.claude_sdk.transient import (
        is_claude_sdk_degenerate_success,
        is_claude_sdk_permanent_api_error,
    )

    cause = ClaudeSDKPermanentAPIError(
        "API Error: 400 `task_budget.total` must be at least 20,000 tokens"
    )
    try:
        raise Exception("Claude Code returned an error result: success") from cause
    except Exception as e:
        assert is_claude_sdk_degenerate_success(e) is True  # transient signature...
        assert is_claude_sdk_permanent_api_error(e) is True  # ...but permanent wins
        assert is_claude_sdk_transient(e) is False


def test_non_permanent_api_error_runtime_error_unaffected():
    from robotsix_llmio.claude_sdk.transient import is_claude_sdk_permanent_api_error

    assert is_claude_sdk_permanent_api_error(RuntimeError("something else")) is False


def test_is_permanent_api_error_text_matches_case_insensitively():
    from robotsix_llmio.claude_sdk.transient import is_permanent_api_error_text

    assert is_permanent_api_error_text("API Error: 400 `task_budget.total` too low")
    assert is_permanent_api_error_text("api error: 400") is True
    assert is_permanent_api_error_text("all good here") is False


def test_retryable_status_codes_stay_transient():
    """Scoped to 400 on purpose — rate limits and server errors must keep
    burning retries, since a re-run genuinely clears them."""
    from robotsix_llmio.claude_sdk.transient import is_permanent_api_error_text

    assert is_permanent_api_error_text("API Error: 429 rate limited") is False
    assert is_permanent_api_error_text("API Error: 500 internal") is False
    assert is_permanent_api_error_text("API Error: 529 overloaded") is False


# --- API 401: dead credential, never retryable at this tier -----------------


def test_auth_error_type_detected_and_not_transient():
    from robotsix_llmio.claude_sdk._errors import ClaudeSDKAuthError
    from robotsix_llmio.claude_sdk.transient import is_claude_sdk_auth_error

    e = ClaudeSDKAuthError("API Error: 401 OAuth access token has expired")
    assert is_claude_sdk_auth_error(e) is True
    # The credential stays dead until a human re-authenticates, so every retry
    # re-sends against the same expired token.
    assert is_claude_sdk_transient(e) is False


def test_auth_error_beats_degenerate_success_wrapper():
    """The live outage shape: an expired OAuth token 401s, the CLI exits
    non-zero, and claude_agent_sdk replaces the real ProcessError with its
    degenerate-success message — which IS transient. The auth cause must win,
    or a plain "re-authenticate" instruction burns 3 retries and surfaces as an
    opaque transport failure."""
    from robotsix_llmio.claude_sdk._errors import ClaudeSDKAuthError
    from robotsix_llmio.claude_sdk.transient import (
        is_claude_sdk_auth_error,
        is_claude_sdk_degenerate_success,
    )

    cause = ClaudeSDKAuthError(
        "Failed to authenticate. API Error: 401 OAuth access token has expired."
    )
    try:
        raise Exception("Claude Code returned an error result: success") from cause
    except Exception as e:
        assert is_claude_sdk_degenerate_success(e) is True  # transient signature...
        assert is_claude_sdk_auth_error(e) is True  # ...but auth wins
        assert is_claude_sdk_transient(e) is False


def test_non_auth_runtime_error_unaffected():
    from robotsix_llmio.claude_sdk.transient import is_claude_sdk_auth_error

    assert is_claude_sdk_auth_error(RuntimeError("something else")) is False


def test_is_auth_error_text_matches_the_cli_wording_case_insensitively():
    from robotsix_llmio.claude_sdk.transient import is_auth_error_text

    # Verbatim wording observed from Claude Code 2.1.199 on an expired token.
    assert is_auth_error_text(
        "Failed to authenticate. API Error: 401 OAuth access token has "
        "expired. Re-authenticate to continue."
    )
    assert is_auth_error_text("api error: 401") is True
    assert is_auth_error_text("OAUTH ACCESS TOKEN HAS EXPIRED") is True
    assert is_auth_error_text("all good here") is False


def test_auth_signature_stays_narrow():
    """Scoped to 401 deliberately: 403 is authorisation rather than a dead
    credential, and 429/5xx are genuinely retryable — none may be swallowed
    into the never-retry class."""
    from robotsix_llmio.claude_sdk.transient import is_auth_error_text

    assert is_auth_error_text("API Error: 403 forbidden") is False
    assert is_auth_error_text("API Error: 429 rate limited") is False
    assert is_auth_error_text("API Error: 500 server error") is False


# ---------------------------------------------------------------------------
# E2BIG spawn refusal — deterministic, so retrying can never clear it.
#
# Observed 2026-08-13 on robotsix-chat: four autonomous sessions each burned
# all three attempts on it inside a ten-second window and surfaced an opaque
# server_error to the user.
# ---------------------------------------------------------------------------


def _e2big() -> OSError:
    """The exact shape subprocess raises when the kernel refuses the exec."""
    return OSError(
        errno.E2BIG,
        "Argument list too long",
        "/usr/local/lib/python3.14/site-packages/claude_agent_sdk/_bundled/claude",
    )


def test_spawn_argv_too_long_is_detected() -> None:
    from robotsix_llmio.claude_sdk.transient import is_claude_sdk_spawn_argv_too_long

    assert is_claude_sdk_spawn_argv_too_long(_e2big()) is True


def test_spawn_argv_too_long_is_detected_through_the_cause_chain() -> None:
    """The SDK wraps it in a transport error before it reaches the retry loop."""
    from robotsix_llmio.claude_sdk.transient import is_claude_sdk_spawn_argv_too_long

    wrapped = RuntimeError("Failed to start Claude Code")
    wrapped.__cause__ = _e2big()
    assert is_claude_sdk_spawn_argv_too_long(wrapped) is True


def test_spawn_argv_too_long_is_not_transient() -> None:
    """The whole point: it must not burn the retry budget."""
    from robotsix_llmio.claude_sdk.transient import is_claude_sdk_transient

    wrapped = RuntimeError("Failed to start Claude Code")
    wrapped.__cause__ = _e2big()
    assert is_claude_sdk_transient(wrapped) is False


def test_other_oserrors_still_retry() -> None:
    """Only E2BIG is excluded — a transient spawn hiccup must still retry."""
    from robotsix_llmio.claude_sdk.transient import is_claude_sdk_spawn_argv_too_long

    again = OSError(errno.EAGAIN, "try again")
    assert is_claude_sdk_spawn_argv_too_long(again) is False
    assert is_claude_sdk_spawn_argv_too_long(RuntimeError("boom")) is False


class TestSessionOptionErrors:
    """A refused session option is decided before the CLI does any work.

    The transport rebuilds an identical command every attempt, so retrying can
    only reproduce it; the caller has to bind a different option. Observed
    2026-08-14 on robotsix-chat, where each of three autonomous sessions burned
    all three attempts on it before the caller's own self-heal got a turn.
    """

    @pytest.mark.parametrize(
        "stderr",
        [
            "No conversation found with session ID: "
            "65d9e053-ccb5-515e-bd5d-d68d8dc94adf",
            "Error: Session ID abc-123 is already in use.",
            "Error: --session-id can only be used with --continue or --resume "
            "if --fork-session is also specified.",
        ],
    )
    def test_session_option_errors_are_not_transient(self, stderr: str) -> None:
        exc = RuntimeError(
            f"Claude Agent SDK transport/process failure (agent): "
            f"Command failed with exit code 1\nCLI stderr:\n{stderr}"
        )
        assert is_claude_sdk_session_option_error(exc) is True
        assert is_claude_sdk_transient(exc) is False

    def test_wins_over_the_process_error_type(self) -> None:
        """ProcessError is normally transient — the session check must run first."""

        class ProcessError(Exception):
            pass

        exc = ProcessError(
            "Command failed with exit code 1\nCLI stderr:\n"
            "No conversation found with session ID: abc"
        )
        assert is_claude_sdk_transient(exc) is False

    def test_an_ordinary_spawn_failure_stays_transient(self) -> None:
        """Guard the exclusion's blast radius: only session wording is excluded."""

        class ProcessError(Exception):
            pass

        exc = ProcessError("Command failed with exit code 1")
        assert is_claude_sdk_session_option_error(exc) is False
        assert is_claude_sdk_transient(exc) is True

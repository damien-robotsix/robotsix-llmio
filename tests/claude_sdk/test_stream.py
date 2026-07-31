"""Tests for ``_stream_query`` in ``robotsix_llmio.claude_sdk._stream``."""

from __future__ import annotations

import asyncio
import sys
from types import SimpleNamespace

import pytest

from robotsix_llmio.claude_sdk._stream import (
    ClaudeSDKActivityEvent,
    _stream_query,
    activity_events,
)

# ---------------------------------------------------------------------------
# _stream_query helpers
# ---------------------------------------------------------------------------


def _install_stream_fake_sdk(monkeypatch) -> SimpleNamespace:
    """Install a minimal fake ``claude_agent_sdk`` for ``_stream_query`` tests
    and return the namespace so callers can set ``fake.query``."""
    fake = SimpleNamespace()

    class _FakeTextBlock:
        def __init__(self, text: str) -> None:
            self.text = text

    class _FakeAssistantMessage:
        def __init__(self, text: str) -> None:
            self.content = [_FakeTextBlock(text)]

    class _FakeResultMessage:
        def __init__(self, result: str | None = None) -> None:
            self.result = result

    # _log_stream_message dispatches on type(message).__name__ (it must work
    # without importing the real SDK classes), so the fakes need the real
    # names, not their Python identifiers.
    _FakeTextBlock.__name__ = "TextBlock"
    _FakeAssistantMessage.__name__ = "AssistantMessage"
    _FakeResultMessage.__name__ = "ResultMessage"

    fake.TextBlock = _FakeTextBlock
    fake.AssistantMessage = _FakeAssistantMessage
    fake.ResultMessage = _FakeResultMessage
    fake.query = None  # to be set by each test

    # Install via monkeypatch ONLY: a preceding raw
    # `sys.modules["claude_agent_sdk"] = fake` would make monkeypatch record
    # the fake as the "original" and restore it (not remove it) on teardown,
    # leaking the incomplete stub into later tests (e.g. test_confine_hook's
    # `from claude_agent_sdk import create_sdk_mcp_server`).
    monkeypatch.setitem(sys.modules, "claude_agent_sdk", fake)
    return fake


# ---------------------------------------------------------------------------
# _stream_query — success / fallback
# ---------------------------------------------------------------------------


def test_stream_query_success(monkeypatch):
    """Mock query() yields AssistantMessage with TextBlock → text returned."""
    fake = _install_stream_fake_sdk(monkeypatch)

    async def _fake_query(*, prompt, options):
        yield fake.AssistantMessage("hello from sdk")
        yield fake.ResultMessage()

    fake.query = _fake_query

    text, result, reasoning = asyncio.run(_stream_query("prompt", None, "test"))
    assert text == "hello from sdk"
    assert isinstance(result, fake.ResultMessage)
    assert reasoning == ""  # no ThinkingBlock streamed → no reasoning


def test_stream_query_result_fallback(monkeypatch):
    """When no text is accumulated from chunks, use ResultMessage.result."""
    fake = _install_stream_fake_sdk(monkeypatch)

    async def _fake_query(*, prompt, options):
        # AssistantMessage with empty text block
        yield fake.AssistantMessage("")
        yield fake.ResultMessage("fallback text")

    fake.query = _fake_query

    text, _result, _reasoning = asyncio.run(_stream_query("prompt", None, "test"))
    assert text == "fallback text"


def test_stream_query_captures_thinking(monkeypatch):
    """``ThinkingBlock`` content is captured and returned as the reasoning,
    separate from the assistant text."""
    fake = _install_stream_fake_sdk(monkeypatch)

    class _FakeThinkingBlock:
        def __init__(self, thinking: str) -> None:
            self.thinking = thinking

    # ``_stream_query`` matches the block by class name, not import.
    _FakeThinkingBlock.__name__ = "ThinkingBlock"

    async def _fake_query(*, prompt, options):
        msg = fake.AssistantMessage("the answer")
        # Both a thinking block and the visible answer in one assistant turn.
        msg.content = [
            _FakeThinkingBlock("step 1\nstep 2"),
            fake.TextBlock("the answer"),
        ]
        yield msg
        yield fake.ResultMessage()

    fake.query = _fake_query

    text, _result, reasoning = asyncio.run(_stream_query("prompt", None, "test"))
    assert text == "the answer"
    assert reasoning == "step 1\nstep 2"


def test_stream_query_blank_thinking_ignored(monkeypatch):
    """A whitespace-only ThinkingBlock contributes no reasoning."""
    fake = _install_stream_fake_sdk(monkeypatch)

    class _FakeThinkingBlock:
        def __init__(self, thinking: str) -> None:
            self.thinking = thinking

    _FakeThinkingBlock.__name__ = "ThinkingBlock"

    async def _fake_query(*, prompt, options):
        msg = fake.AssistantMessage("answer")
        msg.content = [_FakeThinkingBlock("   "), fake.TextBlock("answer")]
        yield msg
        yield fake.ResultMessage()

    fake.query = _fake_query

    _text, _result, reasoning = asyncio.run(_stream_query("prompt", None, "test"))
    assert reasoning == ""


# ---------------------------------------------------------------------------
# _stream_query — usage exhaustion
# ---------------------------------------------------------------------------


def test_stream_query_raises_on_usage_exhausted(monkeypatch):
    """An is_error=True result reporting exhausted credits raises
    ClaudeSDKUsageExhaustedError instead of returning the text as a reply."""
    from robotsix_llmio.claude_sdk.model import ClaudeSDKUsageExhaustedError

    fake = _install_stream_fake_sdk(monkeypatch)

    async def _fake_query(*, prompt, options):
        yield fake.AssistantMessage("You're out of usage credits · resets Jul 9")
        result = fake.ResultMessage()
        result.is_error = True
        yield result

    fake.query = _fake_query

    with pytest.raises(ClaudeSDKUsageExhaustedError) as exc_info:
        asyncio.run(_stream_query("prompt", None, "test"))
    assert "out of usage credits" in str(exc_info.value).lower()


def test_stream_query_is_error_without_usage_signature_unaffected(monkeypatch):
    """An is_error=True result NOT matching the usage-exhaustion wording is
    left exactly as before (out of scope for this fix) — no exception."""
    fake = _install_stream_fake_sdk(monkeypatch)

    async def _fake_query(*, prompt, options):
        yield fake.AssistantMessage("some other error text")
        result = fake.ResultMessage()
        result.is_error = True
        yield result

    fake.query = _fake_query

    text, result, _reasoning = asyncio.run(_stream_query("prompt", None, "test"))
    assert text == "some other error text"
    assert result.is_error is True


def test_stream_query_usage_signature_without_is_error_unaffected(monkeypatch):
    """Matching text with is_error=False (a real, non-error reply that
    happens to mention the phrase) is returned normally — the is_error flag
    gates the check, not the text alone."""
    fake = _install_stream_fake_sdk(monkeypatch)

    async def _fake_query(*, prompt, options):
        yield fake.AssistantMessage("you asked about running out of usage credits")
        yield fake.ResultMessage()  # is_error defaults to falsy/absent

    fake.query = _fake_query

    text, _result, _reasoning = asyncio.run(_stream_query("prompt", None, "test"))
    assert text == "you asked about running out of usage credits"


def test_stream_query_raises_on_usage_exhausted_via_collapsed_exception(monkeypatch):
    """The SDK sometimes collapses usage-exhaustion into a raised generic
    exception (discarding the real text) instead of a clean is_error=True
    return — e.g. the same wording that also matches the degenerate-success
    signature. The text already streamed into `chunks` before that exception
    fired must still be checked, or this leaks to the user as a generic
    "returned an error result: success" error after 3 wasted retries."""
    from robotsix_llmio.claude_sdk.model import ClaudeSDKUsageExhaustedError

    fake = _install_stream_fake_sdk(monkeypatch)

    async def _fake_query(*, prompt, options):
        yield fake.AssistantMessage("You're out of usage credits · resets Jul 9")
        raise Exception("Claude Code returned an error result: success")
        yield  # pragma: no cover — makes this an async generator

    fake.query = _fake_query

    with pytest.raises(ClaudeSDKUsageExhaustedError) as exc_info:
        asyncio.run(_stream_query("prompt", None, "test"))
    assert "out of usage credits" in str(exc_info.value).lower()
    assert isinstance(exc_info.value.__cause__, Exception)


def test_stream_query_collapsed_exception_without_usage_signature_unaffected(
    monkeypatch,
):
    """A collapsed exception whose already-streamed text does NOT mention
    usage exhaustion is left exactly as before (e.g. still handled by the
    existing degenerate-success transient classification upstream)."""
    fake = _install_stream_fake_sdk(monkeypatch)

    async def _fake_query(*, prompt, options):
        yield fake.AssistantMessage("some unrelated partial text")
        raise Exception("Claude Code returned an error result: success")
        yield  # pragma: no cover — makes this an async generator

    fake.query = _fake_query

    with pytest.raises(Exception, match="returned an error result: success"):
        asyncio.run(_stream_query("prompt", None, "test"))


# ---------------------------------------------------------------------------
# _stream_query — timeout
# ---------------------------------------------------------------------------


def test_stream_query_timeout(monkeypatch):
    """asyncio.wait_for timeout raises ClaudeSDKQueryTimeout."""
    from robotsix_llmio.claude_sdk.model import ClaudeSDKQueryTimeout
    from robotsix_llmio.core import constants

    fake = _install_stream_fake_sdk(monkeypatch)

    async def _hanging_query(*, prompt, options):
        await asyncio.sleep(30)
        yield fake.ResultMessage()  # pragma: no cover

    fake.query = _hanging_query
    monkeypatch.setattr(constants, "SDK_QUERY_TIMEOUT", 0.05)

    with pytest.raises(ClaudeSDKQueryTimeout):
        asyncio.run(_stream_query("prompt", None, "test"))


# ---------------------------------------------------------------------------
# _stream_query — extra_transient / turn limit
# ---------------------------------------------------------------------------


def test_stream_query_extra_transient_true(monkeypatch):
    """When extra_transient returns True, original exception is wrapped in
    ClaudeSDKTurnLimitError."""
    from robotsix_llmio.claude_sdk.model import ClaudeSDKTurnLimitError

    fake = _install_stream_fake_sdk(monkeypatch)

    class _Boom(Exception):
        pass

    async def _failing_query(*, prompt, options):
        raise _Boom("hit the cap!")
        yield  # pragma: no cover — makes this an async generator

    fake.query = _failing_query

    def _is_transient(exc: Exception) -> bool:
        return isinstance(exc, _Boom)

    with pytest.raises(ClaudeSDKTurnLimitError) as exc_info:
        asyncio.run(
            _stream_query("prompt", None, "test", extra_transient=_is_transient)
        )
    assert "hit the cap!" in str(exc_info.value.__cause__)


def test_stream_query_extra_transient_false(monkeypatch):
    """When extra_transient returns False, the original exception is wrapped
    in ClaudeSDKAPIError with __cause__ preserving the original."""
    from robotsix_llmio.claude_sdk.model import ClaudeSDKAPIError

    fake = _install_stream_fake_sdk(monkeypatch)

    class _Boom(Exception):
        pass

    async def _failing_query(*, prompt, options):
        raise _Boom("something else")
        yield  # pragma: no cover — makes this an async generator

    fake.query = _failing_query

    def _not_transient(exc: Exception) -> bool:
        return False

    with pytest.raises(ClaudeSDKAPIError) as exc_info:
        asyncio.run(
            _stream_query("prompt", None, "test", extra_transient=_not_transient)
        )
    assert isinstance(exc_info.value.__cause__, _Boom)


def test_stream_query_default_wraps_sdk_error(monkeypatch):
    """When extra_transient is not passed (default None), a raw SDK exception
    is wrapped in ClaudeSDKAPIError, preserving the original as __cause__."""
    from robotsix_llmio.claude_sdk.model import ClaudeSDKAPIError

    fake = _install_stream_fake_sdk(monkeypatch)

    class _RawSDKError(Exception):
        pass

    async def _failing_query(*, prompt, options):
        raise _RawSDKError("transport broken")
        yield  # pragma: no cover — makes this an async generator

    fake.query = _failing_query

    with pytest.raises(ClaudeSDKAPIError) as exc_info:
        asyncio.run(_stream_query("prompt", None, "test"))
    assert isinstance(exc_info.value.__cause__, _RawSDKError)
    assert "transport broken" in str(exc_info.value.__cause__)


# ---------------------------------------------------------------------------
# _stream_query — on_event / activity_events()
# ---------------------------------------------------------------------------


def _fake_query_with_tool_call(fake):
    class _FakeToolUseBlock:
        def __init__(self, name: str, input: dict) -> None:
            self.name = name
            self.input = input

    _FakeToolUseBlock.__name__ = "ToolUseBlock"

    async def _fake_query(*, prompt, options):
        msg = fake.AssistantMessage("")
        msg.content = [_FakeToolUseBlock("search", {"q": "x"})]
        yield msg
        yield fake.AssistantMessage("done")
        yield fake.ResultMessage()

    return _fake_query


def test_stream_query_on_event_explicit_arg(monkeypatch):
    """An explicit on_event= argument receives every streamed activity event."""
    fake = _install_stream_fake_sdk(monkeypatch)
    fake.query = _fake_query_with_tool_call(fake)

    events: list[ClaudeSDKActivityEvent] = []
    text, _result, _reasoning = asyncio.run(
        _stream_query("prompt", None, "test", on_event=events.append)
    )
    assert text == "done"
    kinds = [e.kind for e in events]
    assert kinds == ["tool_call", "text"]


def test_stream_query_no_on_event_and_no_context_is_a_no_op(monkeypatch):
    """With neither an explicit on_event nor an ambient activity_events()
    context, _stream_query behaves exactly as before (no callback invoked,
    no error)."""
    fake = _install_stream_fake_sdk(monkeypatch)
    fake.query = _fake_query_with_tool_call(fake)

    text, _result, _reasoning = asyncio.run(_stream_query("prompt", None, "test"))
    assert text == "done"


def test_stream_query_uses_ambient_activity_events_context(monkeypatch):
    """activity_events() supplies the callback when no explicit on_event is
    passed — the mechanism that lets a caller (e.g. robotsix-chat) receive
    live activity from either the no-tools or tool-loop claude_sdk path
    without threading on_event through build_agent()/run()."""
    fake = _install_stream_fake_sdk(monkeypatch)
    fake.query = _fake_query_with_tool_call(fake)

    events: list[ClaudeSDKActivityEvent] = []
    with activity_events(events.append):
        text, _result, _reasoning = asyncio.run(_stream_query("prompt", None, "test"))
    assert text == "done"
    assert [e.kind for e in events] == ["tool_call", "text"]


def test_stream_query_explicit_on_event_overrides_ambient_context(monkeypatch):
    """An explicit on_event= argument wins over an ambient activity_events()
    context (e.g. a test double vs. the chat-wide default)."""
    fake = _install_stream_fake_sdk(monkeypatch)
    fake.query = _fake_query_with_tool_call(fake)

    ambient_events: list[ClaudeSDKActivityEvent] = []
    explicit_events: list[ClaudeSDKActivityEvent] = []
    with activity_events(ambient_events.append):
        asyncio.run(
            _stream_query("prompt", None, "test", on_event=explicit_events.append)
        )
    assert ambient_events == []
    assert len(explicit_events) == 2


def test_activity_events_context_resets_after_exit(monkeypatch):
    """Once the activity_events() context exits, calls outside it get no
    callback again (the contextvar is reset, not left dangling)."""
    fake = _install_stream_fake_sdk(monkeypatch)
    fake.query = _fake_query_with_tool_call(fake)

    events: list[ClaudeSDKActivityEvent] = []
    with activity_events(events.append):
        asyncio.run(_stream_query("prompt", None, "test"))
    assert len(events) == 2

    events.clear()
    asyncio.run(_stream_query("prompt", None, "test"))  # outside the context
    assert events == []

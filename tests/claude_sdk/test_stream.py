"""Tests for the shared Claude Agent SDK streaming loop in
``robotsix_llmio.claude_sdk._stream`` — covers ``_stream_query``,
``_log_stream_message``, and ``_short``.
"""

from __future__ import annotations

import asyncio
import logging
import sys
from types import SimpleNamespace
from typing import Any

import pytest

from robotsix_llmio.claude_sdk._stream import _log_stream_message, _short, _stream_query

# ---------------------------------------------------------------------------
# _short
# ---------------------------------------------------------------------------


def test_short_string_within_limit():
    assert _short("hello", limit=10) == "hello"


def test_short_string_beyond_limit():
    s = "a" * 250
    result = _short(s, limit=200)
    assert len(result) == 201  # 200 chars + "…"
    assert result.endswith("…")
    assert result.startswith("a" * 200)


def test_short_non_string_uses_json():
    result = _short({"key": "value"}, limit=500)
    assert '"key"' in result
    assert '"value"' in result


def test_short_collapses_whitespace():
    result = _short("hello   world\n\tagain", limit=500)
    assert result == "hello world again"


# ---------------------------------------------------------------------------
# _log_stream_message
# ---------------------------------------------------------------------------


class _FakeMsg:
    """A fake SDK message whose ``type(…).__name__`` matches the real class."""

    def __init__(self, cls_name: str, **attrs: Any) -> None:
        self.__dict__.update(attrs)
        # Give this instance its own anonymous subclass so type(m).__name__
        # returns cls_name without interfering with other instances.
        self.__class__ = type(cls_name, (_FakeMsg,), {})


def _make_msg(cls_name: str, **attrs: Any) -> _FakeMsg:
    """Build a fake SDK message with the given class name and attributes."""
    return _FakeMsg(cls_name, **attrs)


def _make_block(cls_name: str, **attrs: Any) -> _FakeMsg:
    """Build a fake content block."""
    return _FakeMsg(cls_name, **attrs)


def test_log_stream_message_assistant_text(caplog):
    caplog.set_level(logging.INFO, logger="robotsix_llmio.claude_sdk")
    turn = [0]
    block = _make_block("TextBlock", text="hello world")
    msg = _make_msg("AssistantMessage", content=[block])
    _log_stream_message(msg, turn, "test-label")
    assert turn[0] == 1
    assert "test-label turn 1: text — hello world" in caplog.text


def test_log_stream_message_assistant_tool_use(caplog):
    caplog.set_level(logging.INFO, logger="robotsix_llmio.claude_sdk")
    turn = [0]
    block = _make_block("ToolUseBlock", name="search", input={"q": "x"})
    msg = _make_msg("AssistantMessage", content=[block])
    _log_stream_message(msg, turn, "test-label")
    assert turn[0] == 1
    assert "tool_use search(" in caplog.text


def test_log_stream_message_assistant_thinking(caplog):
    caplog.set_level(logging.INFO, logger="robotsix_llmio.claude_sdk")
    turn = [0]
    block = _make_block("ThinkingBlock", thinking="pondering...")
    msg = _make_msg("AssistantMessage", content=[block])
    _log_stream_message(msg, turn, "test-label")
    assert turn[0] == 1
    assert "thinking (12 chars)" in caplog.text


def test_log_stream_message_tool_result(caplog):
    caplog.set_level(logging.INFO, logger="robotsix_llmio.claude_sdk")
    turn = [0]
    block = _make_block("ToolResultBlock", content="done", is_error=False)
    msg = _make_msg("UserMessage", content=[block])
    _log_stream_message(msg, turn, "test-label")
    assert "tool_result — done" in caplog.text


def test_log_stream_message_tool_result_error(caplog):
    caplog.set_level(logging.INFO, logger="robotsix_llmio.claude_sdk")
    turn = [0]
    block = _make_block("ToolResultBlock", content="fail", is_error=True)
    msg = _make_msg("ToolResultMessage", content=[block])
    _log_stream_message(msg, turn, "test-label")
    assert "tool_result [ERROR] — fail" in caplog.text


def test_log_stream_message_result(caplog):
    caplog.set_level(logging.INFO, logger="robotsix_llmio.claude_sdk")
    turn = [5]
    msg = _make_msg(
        "ResultMessage", subtype="success", is_error=False, duration_ms=1234
    )
    _log_stream_message(msg, turn, "test-label")
    expected = "result: subtype=success is_error=False turns=5 duration_ms=1234"
    assert expected in caplog.text


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

    text, result = asyncio.run(_stream_query("prompt", None, "test"))
    assert text == "hello from sdk"
    assert isinstance(result, fake.ResultMessage)


def test_stream_query_result_fallback(monkeypatch):
    """When no text is accumulated from chunks, use ResultMessage.result."""
    fake = _install_stream_fake_sdk(monkeypatch)

    async def _fake_query(*, prompt, options):
        # AssistantMessage with empty text block
        yield fake.AssistantMessage("")
        yield fake.ResultMessage("fallback text")

    fake.query = _fake_query

    text, _result = asyncio.run(_stream_query("prompt", None, "test"))
    assert text == "fallback text"


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
    """When extra_transient returns False, the original exception propagates."""
    fake = _install_stream_fake_sdk(monkeypatch)

    class _Boom(Exception):
        pass

    async def _failing_query(*, prompt, options):
        raise _Boom("something else")
        yield  # pragma: no cover — makes this an async generator

    fake.query = _failing_query

    def _not_transient(exc: Exception) -> bool:
        return False

    with pytest.raises(_Boom):
        asyncio.run(
            _stream_query("prompt", None, "test", extra_transient=_not_transient)
        )

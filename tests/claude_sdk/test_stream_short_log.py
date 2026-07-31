"""Tests for ``_short``, ``_log_stream_message``, and activity event
callbacks in ``robotsix_llmio.claude_sdk._stream``.
"""

from __future__ import annotations

import logging
from typing import Any

from robotsix_llmio.claude_sdk._stream import (
    ClaudeSDKActivityEvent,
    _log_stream_message,
    _short,
)

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
# _log_stream_message — on_event callback
# ---------------------------------------------------------------------------


def test_log_stream_message_emits_tool_call_event():
    events: list[ClaudeSDKActivityEvent] = []
    turn = [0]
    block = _make_block("ToolUseBlock", name="search", input={"q": "x"})
    msg = _make_msg("AssistantMessage", content=[block])
    _log_stream_message(msg, turn, "test-label", events.append)
    assert len(events) == 1
    assert events[0] == ClaudeSDKActivityEvent(
        kind="tool_call", turn=1, tool_name="search", detail='{"q": "x"}'
    )


def test_log_stream_message_emits_tool_result_event():
    events: list[ClaudeSDKActivityEvent] = []
    turn = [1]
    block = _make_block("ToolResultBlock", content="fail", is_error=True)
    msg = _make_msg("ToolResultMessage", content=[block])
    _log_stream_message(msg, turn, "test-label", events.append)
    assert events == [
        ClaudeSDKActivityEvent(kind="tool_result", turn=1, detail="fail", is_error=True)
    ]


def test_log_stream_message_emits_thinking_event():
    events: list[ClaudeSDKActivityEvent] = []
    turn = [0]
    block = _make_block("ThinkingBlock", thinking="pondering...")
    msg = _make_msg("AssistantMessage", content=[block])
    _log_stream_message(msg, turn, "test-label", events.append)
    assert events == [
        ClaudeSDKActivityEvent(kind="thinking", turn=1, detail="12 chars")
    ]


def test_log_stream_message_emits_text_event():
    events: list[ClaudeSDKActivityEvent] = []
    turn = [0]
    block = _make_block("TextBlock", text="hello world")
    msg = _make_msg("AssistantMessage", content=[block])
    _log_stream_message(msg, turn, "test-label", events.append)
    assert events == [ClaudeSDKActivityEvent(kind="text", turn=1, detail="hello world")]


def test_log_stream_message_no_event_for_result_message():
    """ResultMessage has no corresponding activity-event kind (only logged)."""
    events: list[ClaudeSDKActivityEvent] = []
    turn = [1]
    msg = _make_msg("ResultMessage", subtype="success", is_error=False, duration_ms=1)
    _log_stream_message(msg, turn, "test-label", events.append)
    assert events == []


def test_log_stream_message_on_event_none_is_a_no_op():
    """Passing on_event=None (the default) never raises."""
    turn = [0]
    block = _make_block("ToolUseBlock", name="search", input={})
    msg = _make_msg("AssistantMessage", content=[block])
    _log_stream_message(msg, turn, "test-label")  # no on_event given


def test_log_stream_message_broken_callback_does_not_raise(caplog):
    """A callback that raises only drops that event — logging still happens."""
    caplog.set_level(logging.INFO, logger="robotsix_llmio.claude_sdk")
    turn = [0]
    block = _make_block("ToolUseBlock", name="search", input={})
    msg = _make_msg("AssistantMessage", content=[block])

    def _boom(_event: ClaudeSDKActivityEvent) -> None:
        raise RuntimeError("callback exploded")

    _log_stream_message(msg, turn, "test-label", _boom)  # must not raise
    assert "tool_use search(" in caplog.text

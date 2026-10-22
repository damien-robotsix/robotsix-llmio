"""Tests for ``_short``, ``_log_stream_message``, and activity event
callbacks in ``robotsix_llmio.claude_sdk._stream``.
"""

from __future__ import annotations

import logging
from typing import Any

import pytest

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
# credential redaction
# ---------------------------------------------------------------------------

_GIT_REMOTE_V = (
    "origin\thttps://oauth2:ghs_3752211_eyJhbGciOiJSUzI1NiJ9.eyJpc3MiOiIxMjMifQ"
    ".sig-Part_x@github.com/robotsix/repo.git (fetch)\n"
    "origin\thttps://oauth2:ghs_3752211_eyJhbGciOiJSUzI1NiJ9.eyJpc3MiOiIxMjMifQ"
    ".sig-Part_x@github.com/robotsix/repo.git (push)\n"
)


def test_short_redacts_git_remote_oauth2_token():
    result = _short("Exit code 2 " + _GIT_REMOTE_V, limit=500)
    assert "ghs_" not in result
    assert "eyJ" not in result
    assert result == (
        "Exit code 2 "
        "origin https://oauth2:<redacted>@github.com/robotsix/repo.git (fetch) "
        "origin https://oauth2:<redacted>@github.com/robotsix/repo.git (push)"
    )


def test_short_redacts_before_truncating():
    result = _short("x" * 190 + " ghp_" + "A" * 40, limit=200)
    assert "ghp_" not in result
    assert "AAAA" not in result


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("token ghs_abcDEF123 end", "token <redacted> end"),
        ("token ghp_abcDEF123 end", "token <redacted> end"),
        ("pat github_pat_11ABC_def456 end", "pat <redacted> end"),
        (
            "Authorization: Bearer abc.def-ghi_jkl= end",
            "Authorization: Bearer <redacted> end",
        ),
        ("key sk-ant-api03-abcdefgh123 end", "key <redacted> end"),
        ("key sk-or-v1-0123456789abcdef end", "key <redacted> end"),
    ],
)
def test_short_redacts_credential_shapes(raw: str, expected: str):
    assert _short(raw, limit=500) == expected


def test_short_redacts_tool_input_dict():
    result = _short(
        {"command": "curl -H 'Authorization: Bearer ghs_secret123' x"}, limit=500
    )
    assert "secret123" not in result
    assert "Bearer <redacted>" in result


def test_short_leaves_ordinary_text_alone():
    s = "pip install scikit-learn; ask-me; ghost_town; Bearer"
    assert _short(s, limit=500) == s


def test_tool_result_log_and_event_are_redacted(caplog):
    events: list[ClaudeSDKActivityEvent] = []
    block = _make_msg("ToolResultBlock", content=_GIT_REMOTE_V, is_error=True)
    msg = _make_msg("UserMessage", content=[block])
    with caplog.at_level(logging.INFO, logger="robotsix_llmio.claude_sdk"):
        _log_stream_message(msg, [1], "implement", events.append)
    assert "ghs_" not in caplog.text
    assert "oauth2:<redacted>@github.com" in caplog.text
    assert events and "ghs_" not in events[0].detail


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

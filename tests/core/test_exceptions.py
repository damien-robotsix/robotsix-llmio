"""Tests for the base exception hierarchy in ``robotsix_llmio.exceptions``."""

from __future__ import annotations

from robotsix_llmio import RobotsixLLMIOError
from robotsix_llmio.claude_sdk._errors import (
    ClaudeSDKAPIError,
    ClaudeSDKQueryTimeout,
    ClaudeSDKTurnLimitError,
)
from robotsix_llmio.config.loader import TierConfigLoadError


def test_instantiation() -> None:
    """``RobotsixLLMIOError("msg")`` creates an instance without error."""
    err = RobotsixLLMIOError("some message")
    assert err is not None


def test_inherits_from_exception() -> None:
    """``RobotsixLLMIOError`` is a subclass of ``Exception``."""
    assert issubclass(RobotsixLLMIOError, Exception)


def test_subclasses_are_robotsix_errors() -> None:
    """``ClaudeSDKTurnLimitError``, ``ClaudeSDKQueryTimeout``,
    ``ClaudeSDKAPIError``, and ``TierConfigLoadError`` are subclasses of
    ``RobotsixLLMIOError``."""
    assert issubclass(ClaudeSDKTurnLimitError, RobotsixLLMIOError)
    assert issubclass(ClaudeSDKQueryTimeout, RobotsixLLMIOError)
    assert issubclass(ClaudeSDKAPIError, RobotsixLLMIOError)
    assert issubclass(TierConfigLoadError, RobotsixLLMIOError)


def test_message_preservation() -> None:
    """``str(RobotsixLLMIOError("some message"))`` returns the message."""
    assert str(RobotsixLLMIOError("some message")) == "some message"


def test_single_catch_clause() -> None:
    """A single ``except RobotsixLLMIOError`` block catches all four
    exception types."""
    caught: list[str] = []
    for exc in (
        RobotsixLLMIOError("base"),
        ClaudeSDKTurnLimitError("turn limit"),
        ClaudeSDKQueryTimeout("timeout"),
        ClaudeSDKAPIError("api error"),
    ):
        try:
            raise exc
        except RobotsixLLMIOError:
            caught.append(str(exc))
    assert caught == ["base", "turn limit", "timeout", "api error"]


def test_claude_sdk_api_error_preserves_cause() -> None:
    """``ClaudeSDKAPIError`` preserves the original exception as
    ``__cause__`` so the transient classifier can inspect it."""
    original = RuntimeError("simulated SDK failure")
    wrapped = ClaudeSDKAPIError("terminal SDK error")
    wrapped.__cause__ = original
    assert wrapped.__cause__ is original
    assert isinstance(wrapped, RobotsixLLMIOError)

"""Tests for the base exception hierarchy in ``robotsix_llmio.exceptions``."""

from __future__ import annotations

from robotsix_llmio import RobotsixLLMIOError
from robotsix_llmio.claude_sdk.model import (
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
    """``ClaudeSDKTurnLimitError``, ``ClaudeSDKQueryTimeout``, and
    ``TierConfigLoadError`` are subclasses of ``RobotsixLLMIOError``."""
    assert issubclass(ClaudeSDKTurnLimitError, RobotsixLLMIOError)
    assert issubclass(ClaudeSDKQueryTimeout, RobotsixLLMIOError)
    assert issubclass(TierConfigLoadError, RobotsixLLMIOError)


def test_message_preservation() -> None:
    """``str(RobotsixLLMIOError("some message"))`` returns the message."""
    assert str(RobotsixLLMIOError("some message")) == "some message"


def test_single_catch_clause() -> None:
    """A single ``except RobotsixLLMIOError`` block catches all three
    exception types."""
    caught: list[str] = []
    for exc in (
        RobotsixLLMIOError("base"),
        ClaudeSDKTurnLimitError("turn limit"),
        ClaudeSDKQueryTimeout("timeout"),
    ):
        try:
            raise exc
        except RobotsixLLMIOError:
            caught.append(str(exc))
    assert caught == ["base", "turn limit", "timeout"]

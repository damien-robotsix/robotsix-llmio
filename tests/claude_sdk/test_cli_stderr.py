"""Tests for CLI stderr capture.

The SDK's ``ProcessError`` carries only ``Command failed with exit code 1`` and
the placeholder ``stderr='Check stderr output for details'``. The CLI's actual
diagnosis goes to its own stderr, so wrapped errors were unactionable — two
robotsix-chat failures ("Session ID … is already in use", "No conversation
found with session ID: …") were diagnosable only by reading the container log
by hand, even though each names its own fix.
"""

from __future__ import annotations

import asyncio
import contextvars

import pytest

from robotsix_llmio.claude_sdk._cli_stderr import (
    captured,
    describe,
    record,
    start_capture,
)


def test_records_lines_in_order() -> None:
    start_capture()
    record("first")
    record("second")
    assert captured() == "first\nsecond"


def test_ignores_blank_lines() -> None:
    start_capture()
    record("")
    record("   \n")
    record("real")
    assert captured() == "real"


def test_keeps_only_the_most_recent_lines() -> None:
    """The diagnosis is the tail; the buffer rides into an error message."""
    start_capture()
    for i in range(100):
        record(f"line{i}")
    lines = captured().splitlines()
    assert len(lines) <= 20
    assert lines[-1] == "line99"


def test_describe_surfaces_the_real_reason() -> None:
    start_capture()
    record("No conversation found with session ID: abc-123")
    out = describe()
    assert "CLI stderr:" in out
    assert "No conversation found with session ID: abc-123" in out


def test_describe_filters_the_sdk_placeholder() -> None:
    """Echoing 'check stderr output for details' is what made these opaque."""
    start_capture()
    record("Check stderr output for details")
    assert describe() == ""


def test_describe_is_empty_without_capture() -> None:
    """A caller that never started a capture must not get a stray suffix."""
    # A pristine context — copy_context() would inherit a sibling test's buffer.
    assert contextvars.Context().run(describe) == ""


def test_buffers_do_not_leak_across_concurrent_runs() -> None:
    """Agent runs are concurrent; one run's stderr must not explain another's."""

    async def run(tag: str, out: dict[str, str]) -> None:
        start_capture()
        record(f"{tag}-line")
        await asyncio.sleep(0)
        out[tag] = captured()

    async def main() -> dict[str, str]:
        out: dict[str, str] = {}
        await asyncio.gather(run("a", out), run("b", out))
        return out

    result = asyncio.run(main())
    assert result["a"] == "a-line"
    assert result["b"] == "b-line"


class _FailingQuery:
    """Stand-in for ``claude_agent_sdk.query`` that dies the way a bad spawn does."""

    def __init__(self, exc: BaseException) -> None:
        self._exc = exc

    def __call__(self, **_kwargs: object) -> object:
        exc = self._exc

        async def _gen() -> object:
            raise exc
            yield  # pragma: no cover — makes this an async generator

        return _gen()


def test_wrapped_transport_error_carries_the_cli_diagnosis(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The whole point: the raised message must name what the CLI complained about.

    The SDK's ``ProcessError`` says only ``Command failed with exit code 1``
    plus the placeholder stderr, so before this the caller saw nothing
    actionable.
    """
    import claude_agent_sdk

    from robotsix_llmio.claude_sdk import _stream
    from robotsix_llmio.claude_sdk._cli_stderr import record

    process_error = claude_agent_sdk.ProcessError(
        "Command failed with exit code 1",
        exit_code=1,
        stderr="Check stderr output for details",
    )

    def _query(**kwargs: object) -> object:
        # The transport pumps the CLI's stderr into the callback we installed.
        stderr_cb = getattr(kwargs.get("options"), "stderr", None)
        assert stderr_cb is record, "options must carry the capture callback"
        stderr_cb("No conversation found with session ID: abc-123")
        return _FailingQuery(process_error)()

    monkeypatch.setattr(claude_agent_sdk, "query", _query)

    options = claude_agent_sdk.ClaudeAgentOptions(stderr=record)
    with pytest.raises(Exception) as caught:
        asyncio.run(_stream._stream_query("hi", options, "test-label"))

    message = str(caught.value)
    assert "No conversation found with session ID: abc-123" in message

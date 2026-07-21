"""Tests for run_sync / run kwargs: message_history forwarding, unsupported
kwarg warnings, and async message_history threading."""

from __future__ import annotations

import asyncio
import logging

import pytest

pytest.importorskip("pydantic_ai")

from pydantic_ai.messages import (
    ModelRequest,
    ModelResponse,
    TextPart,
    UserPromptPart,
)

from .conftest import (
    _capturing_query,
    _install_fake_sdk,
    _tool_handle,
)

# ---------------------------------------------------------------------------
# run_sync/run kwargs: honor message_history, warn on the rest (never silent)
# ---------------------------------------------------------------------------


def test_tool_run_sync_honors_message_history(monkeypatch):
    """A message_history passed to the tool-loop run_sync is folded into the
    prompt (prior transcript + the new turn), so the caller keeps context."""
    fake = _install_fake_sdk(monkeypatch)
    captured: dict = {}
    fake.query = _capturing_query(fake, captured)

    handle = _tool_handle()
    history = [
        ModelRequest(parts=[UserPromptPart(content="first question")]),
        ModelResponse(parts=[TextPart(content="prior answer")]),
    ]
    handle.run_sync("the new turn", message_history=history)

    prompt = captured["prompt"]
    assert "first question" in prompt
    assert "prior answer" in prompt
    assert prompt.endswith("User: the new turn")  # new turn appended last
    handle.close()


def test_tool_run_sync_without_history_sends_prompt_verbatim(monkeypatch):
    fake = _install_fake_sdk(monkeypatch)
    captured: dict = {}
    fake.query = _capturing_query(fake, captured)

    handle = _tool_handle()
    handle.run_sync("just this")
    assert captured["prompt"] == "just this"  # no history → no transcript wrap
    handle.close()


def test_tool_run_sync_warns_on_unsupported_kwargs(monkeypatch, caplog):
    """Unsupported run kwargs (usage_limits, model_settings) are warned about,
    not silently dropped — and the run still completes."""
    fake = _install_fake_sdk(monkeypatch)
    captured: dict = {}
    fake.query = _capturing_query(fake, captured)

    handle = _tool_handle()
    with caplog.at_level(logging.WARNING, logger="robotsix_llmio.claude_sdk"):
        result = handle.run_sync(
            "hi", usage_limits="L", model_settings={"temperature": 0}
        )

    assert result.output == "done"  # run still works
    warned = " ".join(r.getMessage() for r in caplog.records)
    assert "usage_limits" in warned
    assert "model_settings" in warned
    handle.close()


def test_tool_async_run_honors_message_history(monkeypatch):
    """The async run() path threads message_history through the same way."""
    fake = _install_fake_sdk(monkeypatch)
    captured: dict = {}
    fake.query = _capturing_query(fake, captured)

    handle = _tool_handle()
    history = [ModelRequest(parts=[UserPromptPart(content="earlier ctx")])]
    asyncio.run(handle.run("now", message_history=history))

    assert "earlier ctx" in captured["prompt"]
    assert captured["prompt"].endswith("User: now")
    handle.close()

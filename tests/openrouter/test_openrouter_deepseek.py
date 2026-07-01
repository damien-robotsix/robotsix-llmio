"""Derived DeepSeek layer — pin + per-level reasoning policy."""

from __future__ import annotations

import types
from typing import Any

import pytest


def _model(level: int):
    """Build a DeepSeek model for a capability *level* with reasoning policy
    stamped (as the provider does), without needing network/key beyond
    construction."""
    pytest.importorskip("pydantic_ai.providers.openrouter")
    from robotsix_llmio.openrouter._deepseek_model import OpenRouterDeepseekModel
    from robotsix_llmio.openrouter._deepseek_provider import (
        OpenRouterDeepseekProvider,
    )

    name = {
        1: "deepseek/deepseek-v4-flash",
        2: "deepseek/deepseek-v4-pro",
    }[level]
    from pydantic_ai.providers.openrouter import OpenRouterProvider as _Pyd

    m = OpenRouterDeepseekModel(name, provider=_Pyd(api_key="x"))
    OpenRouterDeepseekProvider(api_key="x")._post_build_model(m, level)
    return m


# --- pin + reasoning policy ------------------------------------------------


def test_level2_pins_and_xhigh():
    m = _model(2)
    ms: dict = {}
    m._inject_pin((), {"model_settings": ms})
    assert ms["extra_body"]["provider"] == {
        "only": ["DeepSeek"],
        "allow_fallbacks": False,
    }
    assert ms["extra_body"]["reasoning"] == {"effort": "xhigh"}


def test_level1_disables_reasoning():
    m = _model(1)
    ms: dict = {}
    m._inject_pin((), {"model_settings": ms})
    assert ms["extra_body"]["provider"]["only"] == ["DeepSeek"]
    assert ms["extra_body"]["reasoning"] == {"enabled": False}


def test_pin_respects_caller_provider_override():
    m = _model(2)
    ms = {"extra_body": {"provider": {"only": ["Other"]}}}
    m._inject_pin((), {"model_settings": ms})
    assert ms["extra_body"]["provider"]["only"] == ["Other"]  # untouched


@pytest.mark.parametrize(
    "level,expected_reasoning",
    [
        (1, {"enabled": False}),
        (2, {"effort": "xhigh"}),
    ],
)
def test_inject_pin_applies_reasoning_even_when_provider_preset(
    level: int, expected_reasoning: dict[str, Any]
):
    """Regression: custom ``extra_body.provider`` must not suppress the
    per-tier reasoning policy (the early return was dropping reasoning)."""
    m = _model(level)
    ms: dict[str, Any] = {"extra_body": {"provider": {"only": ["Other"]}}}
    m._inject_pin((), {"model_settings": ms})
    assert ms["extra_body"]["provider"]["only"] == ["Other"]  # still respected
    assert ms["extra_body"]["reasoning"] == expected_reasoning


# --- _reasoning_text -------------------------------------------------------


def test_reasoning_text_concatenates_only_thinking_parts_in_order():
    """Only ThinkingPart contents are joined, in order, ignoring other parts."""
    pytest.importorskip("pydantic_ai.providers.openrouter")
    from pydantic_ai.messages import TextPart, ThinkingPart

    from robotsix_llmio.openrouter._deepseek_model import _reasoning_text

    message = types.SimpleNamespace(
        parts=[
            ThinkingPart(content="a"),
            TextPart(content="visible"),
            ThinkingPart(content="b"),
        ]
    )
    assert _reasoning_text(message) == "ab"


def test_reasoning_text_returns_empty_without_thinking_parts():
    """A turn with no ThinkingPart yields the empty string."""
    pytest.importorskip("pydantic_ai.providers.openrouter")
    from pydantic_ai.messages import TextPart

    from robotsix_llmio.openrouter._deepseek_model import _reasoning_text

    message = types.SimpleNamespace(parts=[TextPart(content="visible")])
    assert _reasoning_text(message) == ""


def test_reasoning_text_handles_missing_parts():
    """``parts=None`` / no ``parts`` attribute is guarded → empty string."""
    pytest.importorskip("pydantic_ai.providers.openrouter")
    from robotsix_llmio.openrouter._deepseek_model import _reasoning_text

    assert _reasoning_text(types.SimpleNamespace(parts=None)) == ""
    assert _reasoning_text(types.SimpleNamespace()) == ""


def test_reasoning_text_skips_non_str_content():
    """A ThinkingPart whose content is not a str is skipped by the guard."""
    pytest.importorskip("pydantic_ai.providers.openrouter")
    from pydantic_ai.messages import ThinkingPart

    from robotsix_llmio.openrouter._deepseek_model import _reasoning_text

    bad = ThinkingPart(content="x")
    bad.content = None  # type: ignore[assignment]
    message = types.SimpleNamespace(parts=[bad, ThinkingPart(content="y")])
    assert _reasoning_text(message) == "y"


# --- _map_model_response ---------------------------------------------------


def _patch_parent(monkeypatch, canned: Any) -> None:
    """Stub the MRO parent (``OpenAIChatModel._map_model_response``) to return a
    FRESH copy of ``canned`` each call so pop()/assign mutations under test do
    not leak between assertions. A non-dict ``canned`` is returned as-is so the
    non-dict short-circuit branch can be exercised too."""
    pytest.importorskip("pydantic_ai.providers.openrouter")
    from pydantic_ai.models.openai import OpenAIChatModel

    def _fake_parent(self, message):
        return dict(canned) if isinstance(canned, dict) else canned

    monkeypatch.setattr(OpenAIChatModel, "_map_model_response", _fake_parent)


def _thinking_message(*contents: str):
    from pydantic_ai.messages import ThinkingPart

    return types.SimpleNamespace(parts=[ThinkingPart(content=c) for c in contents])


def test_echo_reasoning_property_per_level():
    """``_echo_reasoning`` gates case 6: True on level 2 (capable), False on
    level 1 (reasoning disabled)."""
    assert _model(2)._echo_reasoning is True
    assert _model(1)._echo_reasoning is False


def test_map_model_response_passes_non_assistant_unchanged(monkeypatch):
    """A non-assistant (or non-dict) parent result short-circuits unchanged."""
    m = _model(2)
    _patch_parent(monkeypatch, {"role": "user", "content": "hi"})
    assert m._map_model_response(_thinking_message("t")) == {
        "role": "user",
        "content": "hi",
    }
    # A non-dict parent result also short-circuits unchanged.
    _patch_parent(monkeypatch, ["not", "a", "dict"])
    assert m._map_model_response(_thinking_message("t")) == ["not", "a", "dict"]


def test_map_model_response_always_drops_array_forms(monkeypatch):
    """``reasoning`` / ``reasoning_details`` arrays are dropped on both levels."""
    canned = {
        "role": "assistant",
        "content": "x",
        "reasoning": "r",
        "reasoning_details": [{"type": "thinking"}],
    }
    for level in (2, 1):
        m = _model(level)
        _patch_parent(monkeypatch, canned)
        result = m._map_model_response(_thinking_message())
        assert "reasoning" not in result
        assert "reasoning_details" not in result


def test_map_model_response_level2_stamps_reasoning_content(monkeypatch):
    """Level 2 (capable) + tool_calls → reasoning_content equals the joined text."""
    m = _model(2)
    _patch_parent(
        monkeypatch,
        {"role": "assistant", "tool_calls": [{"id": "1"}]},
    )
    message = _thinking_message("foo", "bar")
    from robotsix_llmio.openrouter._deepseek_model import _reasoning_text

    result = m._map_model_response(message)
    assert result["reasoning_content"] == _reasoning_text(message) == "foobar"


def test_map_model_response_level2_empty_when_no_reasoning(monkeypatch):
    """Level 2 + tool_calls + no ThinkingPart → reasoning_content is an
    empty string (present, NOT popped) — the synthetic/reconstructed turn."""
    m = _model(2)
    _patch_parent(
        monkeypatch,
        {"role": "assistant", "tool_calls": [{"id": "1"}]},
    )
    result = m._map_model_response(_thinking_message())
    assert result["reasoning_content"] == ""


def test_map_model_response_level2_no_tool_calls_strips(monkeypatch):
    """Level 2 without tool_calls → reasoning_content is absent."""
    m = _model(2)
    _patch_parent(
        monkeypatch,
        {"role": "assistant", "content": "x", "reasoning_content": "stale"},
    )
    result = m._map_model_response(_thinking_message("t"))
    assert "reasoning_content" not in result


def test_map_model_response_thinking_only_content_level2(monkeypatch):
    """Level 2: a thinking-only turn (no content, no tool_calls) gets
    ``content`` set to a present string so DeepSeek does not reject it."""
    m = _model(2)
    _patch_parent(monkeypatch, {"role": "assistant"})
    result = m._map_model_response(_thinking_message("Good"))
    assert isinstance(result.get("content"), str)
    assert "tool_calls" not in result


def test_map_model_response_thinking_only_content_level1(monkeypatch):
    """Level 1: a thinking-only turn (no content, no tool_calls) also gets
    ``content`` set to a present string — the guard is independent of
    ``_echo_reasoning``."""
    m = _model(1)
    _patch_parent(monkeypatch, {"role": "assistant"})
    result = m._map_model_response(_thinking_message("Good"))
    assert isinstance(result.get("content"), str)
    assert "tool_calls" not in result


def test_map_model_response_does_not_clobber_real_content(monkeypatch):
    """A turn carrying actual text keeps its content untouched."""
    m = _model(2)
    _patch_parent(monkeypatch, {"role": "assistant", "content": "real text"})
    result = m._map_model_response(_thinking_message("t"))
    assert result["content"] == "real text"


def test_map_model_response_does_not_add_content_to_tool_call_turn(monkeypatch):
    """A tool-call turn with no content stays content-free (tool_calls is valid)."""
    m = _model(2)
    _patch_parent(monkeypatch, {"role": "assistant", "tool_calls": [{"id": "1"}]})
    result = m._map_model_response(_thinking_message("t"))
    assert "content" not in result


def test_map_model_response_level1_strips_with_tool_calls(monkeypatch):
    """Level 1 strips reasoning_content even with tool_calls present."""
    m = _model(1)
    _patch_parent(
        monkeypatch,
        {
            "role": "assistant",
            "tool_calls": [{"id": "1"}],
            "reasoning_content": "stale",
        },
    )
    result = m._map_model_response(_thinking_message("t"))
    assert "reasoning_content" not in result


# --- response key constants ------------------------------------------------


def test_deepseek_response_key_constants_pin_wire_values():
    """The extracted constants must keep their exact DeepSeek wire strings."""
    from robotsix_llmio.openrouter import _deepseek_model as ds_model

    assert ds_model._REASONING_KEY == "reasoning"
    assert ds_model._REASONING_CONTENT_KEY == "reasoning_content"
    assert ds_model._TOOL_CALLS_KEY == "tool_calls"

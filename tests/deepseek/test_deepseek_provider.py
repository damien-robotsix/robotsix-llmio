"""Offline unit tests for the direct-API DeepSeek provider + model.

Fully offline and key-free: key resolution, base URL, the per-level thinking
policy (model-id swap), the absence of OpenRouter-only machinery, and the
``reasoning_content`` round-trip. Each test sets or clears ``DEEPSEEK_API_KEY``
explicitly via ``monkeypatch``.
"""

from __future__ import annotations

import asyncio
import types
from typing import Any

import pytest

# The DeepSeek provider builds an ``AsyncOpenAI`` with the httpx2 client that
# only ``openai>=3`` accepts. Skip the whole module *visibly* (N skipped with a
# reason) when the optional ``openai>=3`` extra is absent or stale, instead of
# silently dropping the directory at collection time.
pytest.importorskip(
    "openai", minversion="3", reason="DeepSeek transport requires openai>=3"
)

from robotsix_llmio.deepseek import DeepseekAPIError
from robotsix_llmio.deepseek.model import (
    NON_THINKING_MODEL,
    THINKING_MODEL,
)
from robotsix_llmio.deepseek.provider import DeepseekProvider


def _model(level: int):
    """Build a DeepSeek model for a capability *level* with the per-level
    thinking policy stamped (as the provider does), without needing network."""
    pytest.importorskip("pydantic_ai.providers.openai")
    from pydantic_ai.providers.openai import OpenAIProvider

    from robotsix_llmio.deepseek.model import DeepseekModel

    name = {1: NON_THINKING_MODEL, 2: THINKING_MODEL}[level]
    m = DeepseekModel(name, provider=OpenAIProvider(api_key="x"))
    DeepseekProvider(api_key="x")._post_build_model(m, level)
    return m


# --- auth resolution -------------------------------------------------------


def test_missing_key_raises(monkeypatch):
    """With no explicit key and no env var, construction raises a clear
    ``DeepseekAPIError`` naming the missing DeepSeek API key."""
    monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)
    with pytest.raises(DeepseekAPIError, match="DeepSeek API key missing"):
        DeepseekProvider(api_key=None)


def test_explicit_api_key_and_default_base_url(monkeypatch):
    """An explicit ``api_key=`` is stored even when the env var is unset, and
    the default ``base_url`` targets the direct DeepSeek API."""
    monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)
    provider = DeepseekProvider(api_key="sk-test")
    assert provider._api_key == "sk-test"
    assert provider._base_url == "https://api.deepseek.com"


def test_env_var_fallback(monkeypatch):
    """When ``api_key`` is ``None`` the constructor falls back to the
    ``DEEPSEEK_API_KEY`` environment variable."""
    monkeypatch.setenv("DEEPSEEK_API_KEY", "sk-env")
    provider = DeepseekProvider(api_key=None)
    assert provider._api_key == "sk-env"


# --- per-level thinking policy (model-id swap) -----------------------------


def test_level1_resolves_non_thinking_model():
    """Level 1 resolves to the non-thinking model (``deepseek-chat``)."""
    assert _model(1).model_name == NON_THINKING_MODEL


def test_level2_resolves_thinking_model():
    """Level >= 2 resolves to the thinking model (``deepseek-reasoner``)."""
    assert _model(2).model_name == THINKING_MODEL


def test_level2_forces_thinking_even_if_chat_configured():
    """A level-2 slot configured with the chat model is forced to the reasoner
    (thinking is selected by model id on the direct API)."""
    from pydantic_ai.providers.openai import OpenAIProvider

    from robotsix_llmio.deepseek.model import DeepseekModel

    m = DeepseekModel(NON_THINKING_MODEL, provider=OpenAIProvider(api_key="x"))
    DeepseekProvider(api_key="x")._post_build_model(m, 2)
    assert m.model_name == THINKING_MODEL


def test_level1_forces_non_thinking_even_if_reasoner_configured():
    """A level-1 slot configured with the reasoner is forced to the chat model."""
    from pydantic_ai.providers.openai import OpenAIProvider

    from robotsix_llmio.deepseek.model import DeepseekModel

    m = DeepseekModel(THINKING_MODEL, provider=OpenAIProvider(api_key="x"))
    DeepseekProvider(api_key="x")._post_build_model(m, 1)
    assert m.model_name == NON_THINKING_MODEL


def test_level0_keeps_configured_model():
    """``level == 0`` (direct ``new_model()``) keeps the configured model id."""
    from pydantic_ai.providers.openai import OpenAIProvider

    from robotsix_llmio.deepseek.model import DeepseekModel

    m = DeepseekModel(THINKING_MODEL, provider=OpenAIProvider(api_key="x"))
    DeepseekProvider(api_key="x")._post_build_model(m, 0)
    assert m.model_name == THINKING_MODEL


# --- no OpenRouter-only machinery ------------------------------------------


def test_model_carries_no_openrouter_routing_state():
    """The direct model has no OpenRouter routing / reasoning-block state and no
    ``_inject_pin`` that would emit it."""
    from robotsix_llmio.deepseek.model import DeepseekModel

    assert not hasattr(DeepseekModel, "_inject_pin")
    assert not hasattr(DeepseekModel, "provider_routing")
    assert not hasattr(DeepseekModel, "reasoning_setting")


def test_completions_create_injects_no_openrouter_machinery(monkeypatch):
    """The direct path adds no provider-routing block, no ``usage.include``, and
    no OpenRouter ``reasoning`` block to the outbound ``model_settings``."""
    from pydantic_ai.models.openai import OpenAIChatModel

    m = _model(2)
    captured: dict[str, Any] = {}

    async def _fake_parent(self, *args, **kwargs):
        captured["kwargs"] = kwargs
        return types.SimpleNamespace(usage=None, model=THINKING_MODEL)

    monkeypatch.setattr(OpenAIChatModel, "_completions_create", _fake_parent)

    settings: dict[str, Any] = {"extra_body": {}}
    result = asyncio.run(m._completions_create(model_settings=settings))

    # model_settings untouched: no provider routing / usage / reasoning added.
    assert settings == {"extra_body": {}}
    assert captured["kwargs"]["model_settings"] == {"extra_body": {}}
    # A non-stream response is returned directly (cost recorded as a side effect).
    assert result.model == THINKING_MODEL


# --- reasoning_content round-trip ------------------------------------------


def _patch_parent(monkeypatch, canned: Any) -> None:
    """Stub the MRO parent (``OpenAIChatModel._map_model_response``) to return a
    FRESH copy of ``canned`` each call so pop()/assign mutations under test do
    not leak between assertions."""
    from pydantic_ai.models.openai import OpenAIChatModel

    def _fake_parent(self, message):
        return dict(canned) if isinstance(canned, dict) else canned

    monkeypatch.setattr(OpenAIChatModel, "_map_model_response", _fake_parent)


def _thinking_message(*contents: str):
    from pydantic_ai.messages import ThinkingPart

    return types.SimpleNamespace(parts=[ThinkingPart(content=c) for c in contents])


def test_echo_reasoning_property_per_level():
    """``_echo_reasoning`` is True on the thinking model, False on the chat one."""
    assert _model(2)._echo_reasoning is True
    assert _model(1)._echo_reasoning is False


def test_map_model_response_thinking_stamps_reasoning_content(monkeypatch):
    """Thinking model + tool_calls → reasoning_content equals the joined text."""
    from robotsix_llmio.deepseek.model import _reasoning_text

    m = _model(2)
    _patch_parent(monkeypatch, {"role": "assistant", "tool_calls": [{"id": "1"}]})
    message = _thinking_message("foo", "bar")
    result = m._map_model_response(message)
    assert result["reasoning_content"] == _reasoning_text(message) == "foobar"


def test_map_model_response_thinking_empty_when_no_reasoning(monkeypatch):
    """Thinking model + tool_calls + no ThinkingPart → reasoning_content is an
    empty string (present, NOT popped) — the synthetic/reconstructed turn."""
    m = _model(2)
    _patch_parent(monkeypatch, {"role": "assistant", "tool_calls": [{"id": "1"}]})
    result = m._map_model_response(_thinking_message())
    assert result["reasoning_content"] == ""


def test_map_model_response_non_thinking_strips_with_tool_calls(monkeypatch):
    """The non-thinking (chat) model strips reasoning_content even with
    tool_calls present."""
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


def test_map_model_response_always_drops_array_forms(monkeypatch):
    """``reasoning`` / ``reasoning_details`` arrays are dropped on both models."""
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


def test_map_model_response_thinking_only_turn_gets_empty_content(monkeypatch):
    """A thinking-only turn (no content, no tool_calls) gets ``content`` set to a
    present string so DeepSeek does not reject it."""
    m = _model(2)
    _patch_parent(monkeypatch, {"role": "assistant"})
    result = m._map_model_response(_thinking_message("Good"))
    assert isinstance(result.get("content"), str)
    assert "tool_calls" not in result

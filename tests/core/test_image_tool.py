"""Unit tests for the image-question tool and its build_agent wiring."""

from __future__ import annotations

import asyncio
from typing import Any
from unittest.mock import MagicMock, patch

from robotsix_llmio.config.tier import TierConfig, TierLevelConfig
from robotsix_llmio.core.image_tool import (
    _MAX_ANSWER_CHARS,
    IMAGE_NOTE_TEMPLATE,
    _augment_with_image_tool,
    build_image_question_tool,
)

_IMAGES = [("image/png", b"png-bytes"), ("image/jpeg", b"jpeg-bytes")]


# --------------------------------------------------------------------------- #
#  Tool behaviour                                                             #
# --------------------------------------------------------------------------- #


def test_out_of_range_index_returns_error_string_not_exception():
    tool = build_image_question_tool(_IMAGES)
    for bad in (-1, 2, 99):
        answer = asyncio.run(tool(bad, "what is this?"))
        assert "out of range" in answer
        assert "2 image(s)" in answer


def test_question_and_image_are_forwarded_to_the_vision_model():
    seen: dict[str, Any] = {}

    async def fake_vision(vision_tlc, media_type, data, question, *, api_key):
        seen.update(
            model=vision_tlc.model,
            media_type=media_type,
            data=data,
            question=question,
            api_key=api_key,
        )
        return "a red button"

    tool = build_image_question_tool(_IMAGES, api_key="sk-vision")
    with patch(
        "robotsix_llmio.core.image_tool._ask_vision_model", side_effect=fake_vision
    ):
        answer = asyncio.run(tool(1, "what colour is the button?"))

    assert answer == "a red button"
    assert seen["media_type"] == "image/jpeg"
    assert seen["data"] == b"jpeg-bytes"
    assert seen["question"] == "what colour is the button?"
    assert seen["api_key"] == "sk-vision"
    assert seen["model"] == "openrouter-deepseek/deepseek-v4-flash-vision-exp"


def test_custom_vision_binding_is_used():
    captured: dict[str, Any] = {}

    async def fake_vision(vision_tlc, *a, **kw):
        captured["model"] = vision_tlc.model
        return "ok"

    cfg = TierConfig(vision=TierLevelConfig(model="openrouter-google/gemini-2-flash"))
    tool = build_image_question_tool(_IMAGES, tier_config=cfg)
    with patch(
        "robotsix_llmio.core.image_tool._ask_vision_model", side_effect=fake_vision
    ):
        asyncio.run(tool(0, "?"))
    assert captured["model"] == "openrouter-google/gemini-2-flash"


def test_vision_failure_returns_explanatory_string():
    async def boom(*a, **kw):
        raise RuntimeError("no endpoints")

    tool = build_image_question_tool(_IMAGES)
    with patch("robotsix_llmio.core.image_tool._ask_vision_model", side_effect=boom):
        answer = asyncio.run(tool(0, "?"))
    assert "ask_image error" in answer
    assert "RuntimeError" in answer


def test_long_answers_are_truncated():
    async def rambling(*a, **kw):
        return "x" * (_MAX_ANSWER_CHARS + 500)

    tool = build_image_question_tool(_IMAGES)
    with patch(
        "robotsix_llmio.core.image_tool._ask_vision_model", side_effect=rambling
    ):
        answer = asyncio.run(tool(0, "?"))
    assert len(answer) < _MAX_ANSWER_CHARS + 50
    assert answer.endswith("[truncated]")


# --------------------------------------------------------------------------- #
#  _augment_with_image_tool                                                   #
# --------------------------------------------------------------------------- #


def test_augment_appends_tool_and_prompt_note():
    def existing_tool() -> str:
        return "x"

    prompt, tools = _augment_with_image_tool(
        "Base prompt.", [existing_tool], _IMAGES, None, None
    )
    assert prompt.startswith("Base prompt.")
    assert IMAGE_NOTE_TEMPLATE.format(n=2, max_index=1) in prompt
    assert tools[0] is existing_tool
    assert tools[1].__name__ == "ask_image"


# --------------------------------------------------------------------------- #
#  build_agent wiring — generic (pydantic-ai) path                            #
# --------------------------------------------------------------------------- #


class _FakeProvider:
    """Minimal LLMProvider with a mocked model constructor."""

    def __init__(self) -> None:
        from robotsix_llmio.core.provider import LLMProvider

        class _P(LLMProvider):
            def new_model(self, *, model=None, level=0):
                return MagicMock(), None

        self.provider = _P()


def test_core_build_agent_wires_ask_image():
    captured: dict[str, Any] = {}

    def fake_build_agent(model, http_client, *, system_prompt, tools, **kw):
        captured["system_prompt"] = system_prompt
        captured["tools"] = tools
        return MagicMock()

    provider = _FakeProvider().provider
    with patch("robotsix_llmio.core.provider._build_agent", fake_build_agent):
        provider.build_agent(
            level=2,
            system_prompt="Do things.",
            images=_IMAGES,
            vision_api_key="sk-v",
        )

    assert "ask_image" in [getattr(t, "__name__", "") for t in captured["tools"]]
    assert "2 image(s) are attached" in captured["system_prompt"]


def test_core_build_agent_without_images_adds_nothing():
    captured: dict[str, Any] = {}

    def fake_build_agent(model, http_client, *, system_prompt, tools, **kw):
        captured["tools"] = tools
        captured["system_prompt"] = system_prompt
        return MagicMock()

    provider = _FakeProvider().provider
    with patch("robotsix_llmio.core.provider._build_agent", fake_build_agent):
        provider.build_agent(level=2, system_prompt="Do things.")

    assert captured["tools"] is None or captured["tools"] == []
    assert "attached" not in captured["system_prompt"]


def test_openrouter_provider_key_reused_for_vision(monkeypatch):
    """On an OpenRouter-family provider the vision tool inherits the
    provider's own api_key unless vision_api_key overrides it."""
    from robotsix_llmio.openrouter._deepseek_provider import OpenRouterDeepseekProvider

    captured: dict[str, Any] = {}

    def fake_augment(system_prompt, tools, images, tier_config, api_key):
        captured["api_key"] = api_key
        return system_prompt, list(tools or [])

    provider = OpenRouterDeepseekProvider(api_key="sk-provider")
    with (
        patch("robotsix_llmio.core.image_tool._augment_with_image_tool", fake_augment),
        patch("robotsix_llmio.core.provider._build_agent", lambda *a, **k: MagicMock()),
        patch.object(
            type(provider), "new_model", lambda self, **kw: (MagicMock(), None)
        ),
    ):
        provider.build_agent(level=2, system_prompt="p", images=_IMAGES)
    assert captured["api_key"] == "sk-provider"


# --------------------------------------------------------------------------- #
#  build_agent wiring — Claude SDK path                                       #
# --------------------------------------------------------------------------- #


def test_claude_sdk_serves_images_natively_no_tool_path():
    """Claude models read images natively: build_agent(images=...) feeds the
    native image flow and injects NO ask_image tool and NO prompt note."""
    from robotsix_llmio.claude_sdk.provider import ClaudeSDKProvider

    provider = ClaudeSDKProvider()
    with patch(
        "robotsix_llmio.core.provider._build_agent",
        lambda model, http_client, **kw: MagicMock(model=model, kw=kw),
    ):
        handle = provider.build_agent(
            level=2,
            system_prompt="Look at the attachment.",
            images=_IMAGES,
        )
    # No-tools path: a real ClaudeSDKModel was built and carries the
    # attachments for its native SDK image blocks.
    assert handle.model.extra_images == _IMAGES
    assert handle.kw["tools"] is None or handle.kw["tools"] == []
    assert "attached" not in handle.kw["system_prompt"]


def test_claude_sdk_tool_path_carries_images_natively():
    from robotsix_llmio.claude_sdk.provider import ClaudeSDKProvider

    def a_tool() -> str:
        return "x"

    provider = ClaudeSDKProvider()
    handle = provider.build_agent(
        level=2,
        system_prompt="Look at the attachment.",
        tools=[a_tool],
        images=_IMAGES,
    )
    assert handle._extra_images == _IMAGES
    # Native transport — no ask_image tool, no prompt note.
    assert not any("ask_image" in t for t in handle._allowed_tools)
    assert "attached" not in handle._system_prompt


def test_claude_sdk_build_agent_no_images_keeps_no_tools_path():
    from robotsix_llmio.claude_sdk.provider import ClaudeSDKProvider

    provider = ClaudeSDKProvider()
    with (
        patch.object(
            type(provider), "new_model", return_value=(MagicMock(), None)
        ) as nm,
        patch(
            "robotsix_llmio.core.provider._build_agent",
            lambda *a, **k: MagicMock(),
        ),
    ):
        provider.build_agent(level=2, system_prompt="p")
        assert nm.called

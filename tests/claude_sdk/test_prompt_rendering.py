"""Prompt rendering, system prompt assembly, and native image support tests
extracted from test_claude_sdk.py."""

from __future__ import annotations

import pytest

pytest.importorskip("pydantic_ai")

from pydantic_ai.messages import (
    BinaryContent,
    ModelRequest,
    ModelResponse,
    RetryPromptPart,
    SystemPromptPart,
    TextPart,
    ToolReturnPart,
    UserPromptPart,
)
from pydantic_ai.models import ModelRequestParameters

from robotsix_llmio.claude_sdk._model import ClaudeSDKModel
from robotsix_llmio.claude_sdk._prompt import (
    build_sdk_prompt,
    collect_latest_user_images,
    extract_prompt_parts,
    render_prompt,
)
from robotsix_llmio.claude_sdk._tool_agent import _SdkToolAgentHandle

# --- prompt rendering ------------------------------------------------------


def test_single_user_turn_sent_verbatim():
    msgs = [ModelRequest(parts=[UserPromptPart(content="hello there")])]
    assert render_prompt(msgs) == "hello there"


def test_multi_turn_rendered_as_transcript():
    msgs = [
        ModelRequest(parts=[UserPromptPart(content="first")]),
        ModelResponse(parts=[TextPart(content="bad json")]),
        ModelRequest(parts=[RetryPromptPart(content="invalid, retry", tool_name=None)]),
    ]
    out = render_prompt(msgs)
    assert "User: first" in out
    assert "Assistant: bad json" in out
    assert "User:" in out.split("Assistant: bad json")[1]  # retry rendered last


def test_tool_return_part_rendered_as_user_text():
    msgs = [
        ModelRequest(
            parts=[ToolReturnPart(tool_name="lookup", content="42", tool_call_id="c1")]
        )
    ]
    assert "Tool result (lookup): 42" in render_prompt(msgs)


def test_binary_content_rendered_as_placeholder_not_bytes():
    """A BinaryContent part (e.g. an attached image) must flatten to a compact
    placeholder — ``str()`` on it reprs the raw bytes, ballooning a megabyte
    image into a multi-megabyte prompt that stalls the CLI subprocess."""

    image = BinaryContent(
        data=b"\x89PNG" + bytes(range(256)) * 64, media_type="image/png"
    )
    msgs = [
        ModelRequest(parts=[UserPromptPart(content=["look at this picture", image])])
    ]

    out = render_prompt(msgs)
    assert "look at this picture" in out
    assert "[binary attachment: image/png," in out
    assert "not visible" in out
    # The rendered prompt must stay small — no escaped-byte blow-up.
    assert len(out) < 500
    assert "\\x89" not in out


def test_bare_binary_content_rendered_as_placeholder():
    """A lone BinaryContent (not wrapped in a list) is also placeholdered."""

    image = BinaryContent(data=bytes(1024), media_type="image/jpeg")
    msgs = [ModelRequest(parts=[UserPromptPart(content=[image])])]

    out = render_prompt(msgs)
    assert out == (
        "[binary attachment: image/jpeg, 1024 bytes — "
        "not visible to this text-only model]"
    )


# --- system prompt assembly ------------------------------------------------


def _params(output_mode="text"):
    return ModelRequestParameters(output_mode=output_mode)


def test_system_text_combines_instructions_and_system_parts():
    m = ClaudeSDKModel("opus")
    msgs = [
        ModelRequest(
            parts=[
                SystemPromptPart(content="be terse"),
                UserPromptPart(content="hi"),
            ],
            instructions="answer in french",
        )
    ]
    sys = m._system_text(msgs, _params())
    assert "be terse" in sys and "answer in french" in sys


# --- native image support ---------------------------------------------------


def test_extract_prompt_parts_splits_text_and_images():

    from robotsix_llmio.claude_sdk._prompt import extract_prompt_parts

    image = BinaryContent(data=b"\x89PNG12345", media_type="image/png")
    audio = BinaryContent(data=b"RIFF1234", media_type="audio/wav")
    text, images = extract_prompt_parts(["describe this", image, audio])

    assert "describe this" in text
    # Non-image binary degrades to the placeholder, image is extracted.
    assert "[binary attachment: audio/wav," in text
    assert images == [("image/png", b"\x89PNG12345")]


def test_extract_prompt_parts_plain_string_passthrough():

    assert extract_prompt_parts("just text") == ("just text", [])


def test_build_sdk_prompt_without_images_is_text():

    assert build_sdk_prompt("hello", []) == "hello"


def test_build_sdk_prompt_with_images_builds_streaming_input():
    import base64

    raw = b"\x89PNGfake"
    messages = build_sdk_prompt("look", [("image/png", raw)])

    assert isinstance(messages, list) and len(messages) == 1
    content = messages[0]["message"]["content"]
    assert content[0] == {"type": "text", "text": "look"}
    assert content[1]["type"] == "image"
    assert content[1]["source"]["media_type"] == "image/png"
    assert base64.b64decode(content[1]["source"]["data"]) == raw


def test_tool_agent_prepare_prompt_extracts_images_no_byte_dump():
    """Regression: the tool path stringified list prompts via an f-string,
    ballooning an attached image into a multi-megabyte escaped-byte prompt."""
    from pydantic_ai.messages import ModelRequest, UserPromptPart

    image = BinaryContent(data=bytes(100_000), media_type="image/png")
    handle = _SdkToolAgentHandle(
        sdk_model="sonnet", system_prompt="sys", server=None, allowed_tools=[]
    )
    history = [ModelRequest(parts=[UserPromptPart(content="earlier turn")])]

    prompt, _system, images = handle._prepare_prompt(["see image", image], history)

    assert "earlier turn" in prompt
    assert "see image" in prompt
    assert len(prompt) < 1000  # no escaped-byte blow-up
    assert images == [("image/png", bytes(100_000))]


def test_collect_latest_user_images_only_newest_turn():
    from pydantic_ai.messages import ModelRequest, UserPromptPart

    old_image = BinaryContent(data=b"old", media_type="image/png")
    new_image = BinaryContent(data=b"new", media_type="image/jpeg")
    msgs = [
        ModelRequest(parts=[UserPromptPart(content=["first", old_image])]),
        ModelRequest(parts=[UserPromptPart(content=["second", new_image])]),
    ]

    assert collect_latest_user_images(msgs) == [("image/jpeg", b"new")]
    assert (
        collect_latest_user_images(
            [ModelRequest(parts=[UserPromptPart(content="text only")])]
        )
        == []
    )


def test_prepare_prompt_tool_naming_note_with_tools():
    """With bridged tools, the system prompt states the mcp__milltools__ prefix
    rule (2026-09-01: haiku called bare `component_request`/`complete_subsession`
    from skill text and burned tool-error turns)."""
    from .conftest import _make_minimal_handle

    handle = _make_minimal_handle(
        allowed_tools=["mcp__milltools__ticket_poll"],
    )
    _, system_prompt, _ = handle._prepare_prompt("hello", None)
    assert "mcp__milltools__" in system_prompt
    assert "bare name" in system_prompt


def test_prepare_prompt_no_naming_note_without_tools():
    """No injected tools → no naming note polluting the system prompt."""
    from .conftest import _make_minimal_handle

    handle = _make_minimal_handle()
    _, system_prompt, _ = handle._prepare_prompt("hello", None)
    assert "mcp__milltools__" not in system_prompt

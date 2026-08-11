"""Tests for structured-output JSON extraction, multi-output parsing,
and build_agent output_type wrapping — extracted from test_claude_sdk.py."""

from __future__ import annotations

import pytest

pytest.importorskip("pydantic_ai")

from pydantic_ai import PromptedOutput
from pydantic_ai.tools import Tool as PydanticTool

from robotsix_llmio.claude_sdk._output import _extract_json_object, _parse_output
from robotsix_llmio.claude_sdk.provider import ClaudeSDKProvider

from .conftest import (
    _HAIKU_AT_LEVEL1,
    _OPUS_AT_LEVEL2,
    _AltVerdict,
    _echo_sync,
    _install_fake_sdk,
    _make_minimal_handle,
    _Verdict,
)

# ---------------------------------------------------------------------------
# structured-output JSON extraction (prose + fenced / stray braces)
# ---------------------------------------------------------------------------


def test_parse_output_str_passthrough():
    assert _parse_output("anything", str) == "anything"


def test_extract_clean_json():
    assert _extract_json_object('{"verdict": "APPROVE"}') == {"verdict": "APPROVE"}


def test_extract_fenced_json_after_prose():
    # The 402b shape: prose preamble, then a ```json fence with the verdict.
    text = (
        "Looking at this review.\n\n## Analysis\nlooks good.\n\n"
        '```json\n{"verdict": "APPROVE", "auto_merge_eligible": true}\n```\n'
    )
    v = _parse_output(text, _Verdict)
    assert isinstance(v, _Verdict)
    assert v.verdict == "APPROVE" and v.auto_merge_eligible is True


def test_extract_ignores_stray_prose_brace():
    # A stray `{...}` in prose must NOT derail extraction of the real object
    # (the old greedy re.search anchored on the first brace and failed).
    text = (
        "The `{verified_proposals}` kwarg is passed through. Verdict below:\n"
        '```json\n{"verdict": "REQUEST_CHANGES"}\n```'
    )
    v = _parse_output(text, _Verdict)
    assert v.verdict == "REQUEST_CHANGES"


def test_extract_prose_wrapped_json_no_fence():
    # No fence, just prose then a JSON object with nested structures.
    text = (
        'Here is my verdict: {"verdict": "APPROVE", "auto_merge_eligible": false} done.'
    )
    v = _parse_output(text, _Verdict)
    assert v.verdict == "APPROVE"


def test_extract_picks_last_valid_object():
    # An earlier non-matching object (e.g. an example) then the real one.
    text = (
        'Example shape: {"foo": 1}\n\nActual:\n'
        '```json\n{"verdict": "NEEDS_DISCUSSION"}\n```'
    )
    v = _parse_output(text, _Verdict)
    assert v.verdict == "NEEDS_DISCUSSION"


def test_extract_no_json_falls_back_to_text():
    import pytest

    with pytest.raises(ValueError, match="no JSON object found"):
        _parse_output("no json at all here", _Verdict)


def test_extract_nested_object_captured_whole():
    text = '```json\n{"verdict": "APPROVE", "nested": {"a": {"b": [1,2]}}}\n```'
    assert _extract_json_object(text) == {
        "verdict": "APPROVE",
        "nested": {"a": {"b": [1, 2]}},
    }


# ---------------------------------------------------------------------------
# multi-output structured output (PromptedOutput)
# ---------------------------------------------------------------------------


def test_parse_output_multi_output_first_type_matches():
    """A-shaped JSON validates against the first model in PromptedOutput."""

    text = '{"verdict": "APPROVE", "auto_merge_eligible": false}'
    result = _parse_output(text, PromptedOutput([_Verdict, _AltVerdict]))
    assert isinstance(result, _Verdict)
    assert result.verdict == "APPROVE"


def test_parse_output_multi_output_second_type_matches():
    """B-shaped JSON falls through to the second model in PromptedOutput."""

    text = '{"outcome": "rejected"}'
    result = _parse_output(text, PromptedOutput([_Verdict, _AltVerdict]))
    assert isinstance(result, _AltVerdict)
    assert result.outcome == "rejected"


def test_parse_output_empty_prompted_output_returns_raw_data():
    """PromptedOutput([]) produces no validators — raw dict is returned."""
    text = '{"verdict": "APPROVE", "auto_merge_eligible": false}'
    result = _parse_output(text, PromptedOutput([]))
    assert result == {"verdict": "APPROVE", "auto_merge_eligible": False}


def test_parse_output_multi_output_no_match_raises():
    """JSON matching no declared model raises (not silently returns str)."""
    import pytest
    from pydantic import ValidationError

    text = '{"unrecognised_field": 42}'
    with pytest.raises(ValidationError):
        _parse_output(text, PromptedOutput([_Verdict, _AltVerdict]))


def test_prepare_prompt_multi_output_anyof_schema():
    """System prompt contains anyOf when PromptedOutput wraps multiple types."""

    handle = _make_minimal_handle(output_type=PromptedOutput([_Verdict, _AltVerdict]))
    _, system_prompt, _ = handle._prepare_prompt("hello", None)
    assert "anyOf" in system_prompt


def test_prepare_prompt_single_prompted_output_no_anyof():
    """Single-model PromptedOutput uses a flat schema (no anyOf)."""

    handle = _make_minimal_handle(output_type=PromptedOutput(_Verdict))
    _, system_prompt, _ = handle._prepare_prompt("hello", None)
    assert "anyOf" not in system_prompt
    assert "verdict" in system_prompt


def test_sdk_tool_agent_handle_rejects_list_output_type():
    """Bare list output_type raises UserError at construction time."""
    import pytest
    from pydantic_ai.exceptions import UserError

    with pytest.raises(UserError, match="list/union output_type is not supported"):
        _make_minimal_handle(output_type=[_Verdict, _AltVerdict])


# ---------------------------------------------------------------------------
# build_agent output_type wrapping for claude-sdk no-tools path
# ---------------------------------------------------------------------------


def test_build_agent_level1_raw_model_wrapped_in_prompted_output():
    """At level=1 with a raw pydantic model, the no-tools path wraps it in
    PromptedOutput before delegating to super(), so ClaudeSDKModel does not
    reject it."""
    provider = ClaudeSDKProvider()
    handle = provider.build_agent(
        level=1,
        tier_config=_HAIKU_AT_LEVEL1,
        system_prompt="You are helpful.",
        output_type=_Verdict,
        tools=None,
    )

    assert isinstance(handle._agent.output_type, PromptedOutput)  # type: ignore[attr-defined]
    unwrapped = handle._agent.output_type.outputs  # type: ignore[attr-defined]
    assert unwrapped is _Verdict
    handle.close()


def test_build_agent_level1_str_output_type_unchanged():
    """str output_type at level=1 is not wrapped (already a valid type)."""
    provider = ClaudeSDKProvider()
    handle = provider.build_agent(
        level=1,
        tier_config=_HAIKU_AT_LEVEL1,
        system_prompt="You are helpful.",
        output_type=str,
        tools=[],
    )
    assert handle._agent.output_type is str  # type: ignore[attr-defined]
    handle.close()


def test_build_agent_level1_already_wrapped_no_double_wrap():
    """When the caller passes PromptedOutput explicitly at level=1, it is not
    double-wrapped."""

    provider = ClaudeSDKProvider()
    handle = provider.build_agent(
        level=1,
        tier_config=_HAIKU_AT_LEVEL1,
        system_prompt="You are helpful.",
        output_type=PromptedOutput(_Verdict),
        tools=None,
    )
    # Should still be a single PromptedOutput wrapping _Verdict.
    assert isinstance(handle._agent.output_type, PromptedOutput)  # type: ignore[attr-defined]
    assert handle._agent.output_type.outputs is _Verdict  # type: ignore[attr-defined]
    handle.close()


def test_build_agent_level2_raw_model_still_works():
    """At level=2 the local wrap applies (output_type is not yet a marker),
    then _resolve_output_type sees PromptedOutput and passes it through —
    no double-wrap, and the agent builds correctly."""
    provider = ClaudeSDKProvider()
    handle = provider.build_agent(
        level=2,
        tier_config=_OPUS_AT_LEVEL2,
        system_prompt="You are helpful.",
        output_type=_Verdict,
        tools=None,
    )

    assert isinstance(handle._agent.output_type, PromptedOutput)  # type: ignore[attr-defined]
    assert handle._agent.output_type.outputs is _Verdict  # type: ignore[attr-defined]
    handle.close()


def test_build_agent_tool_path_output_type_unaffected(monkeypatch):
    """The tool path passes output_type through to _SdkToolAgentHandle
    unchanged — the local wrap only runs on the no-tools branch."""
    fake = _install_fake_sdk(monkeypatch)

    async def _fake_query(*, prompt, options):
        yield fake.AssistantMessage("done")
        yield fake.ResultMessage({"input_tokens": 1, "output_tokens": 1})

    fake.query = _fake_query

    provider = ClaudeSDKProvider()
    handle = provider.build_agent(
        level=1,
        tier_config=_HAIKU_AT_LEVEL1,
        system_prompt="sys",
        tools=[PydanticTool(_echo_sync, name="echo_sync")],
        output_type=_Verdict,
    )
    # The tool path stores output_type directly — no PromptedOutput wrap.
    assert handle._output_type is _Verdict  # type: ignore[attr-defined]
    handle.close()

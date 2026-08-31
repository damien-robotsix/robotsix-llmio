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
        1: "deepseek/deepseek-v4-flash-latest",
        2: "deepseek/deepseek-v4-pro",
    }[level]
    from pydantic_ai.providers.openrouter import OpenRouterProvider as _Pyd

    m = OpenRouterDeepseekModel(name, provider=_Pyd(api_key="x"))
    OpenRouterDeepseekProvider(api_key="x")._post_build_model(m, level)
    return m


# --- pin + reasoning policy ------------------------------------------------


def test_level2_prefers_deepseek_and_xhigh():
    from robotsix_llmio.openrouter._deepseek_model import (
        DEFAULT_IGNORE_CAPABLE,
        DEFAULT_MAX_PRICE_CAPABLE,
    )

    m = _model(2)
    ms: dict = {}
    m._inject_pin((), {"model_settings": ms})
    assert ms["extra_body"]["provider"] == {
        "allow_fallbacks": True,
        "order": ["DeepSeek"],
        "max_price": DEFAULT_MAX_PRICE_CAPABLE,
        "ignore": list(DEFAULT_IGNORE_CAPABLE),
    }
    assert ms["extra_body"]["reasoning"] == {"effort": "xhigh"}


def test_level1_disables_reasoning_and_uses_cheap_ceiling():
    from robotsix_llmio.openrouter._deepseek_model import DEFAULT_MAX_PRICE_CHEAP

    m = _model(1)
    ms: dict = {}
    m._inject_pin((), {"model_settings": ms})
    assert ms["extra_body"]["provider"]["order"] == ["DeepSeek"]
    assert ms["extra_body"]["provider"]["max_price"] == DEFAULT_MAX_PRICE_CHEAP
    assert ms["extra_body"]["reasoning"] == {"enabled": False}


def test_level1_baked_default_prefers_cheap_deepinfra_not_deepseek():
    """The baked cheap-tier binding (``LEVEL1_DEFAULT.provider_kwargs``) routes
    to a stable cheap upstream (DeepInfra) under the cheap ceiling — not
    DeepSeek, whose repriced flash endpoint (~$0.44/$1.32 on 2026-08-31) breaks
    that ceiling and tripped the price-ceiling drift guard."""
    pytest.importorskip("pydantic_ai.providers.openrouter")
    from pydantic_ai.providers.openrouter import OpenRouterProvider as _Pyd

    from robotsix_llmio.config.tier import LEVEL1_DEFAULT
    from robotsix_llmio.openrouter._deepseek_model import (
        DEFAULT_MAX_PRICE_CHEAP,
        OpenRouterDeepseekModel,
    )
    from robotsix_llmio.openrouter._deepseek_provider import (
        OpenRouterDeepseekProvider,
    )

    m = OpenRouterDeepseekModel(
        "deepseek/deepseek-v4-flash-20260731", provider=_Pyd(api_key="x")
    )
    OpenRouterDeepseekProvider(
        api_key="x", **LEVEL1_DEFAULT.provider_kwargs
    )._post_build_model(m, 1)
    ms: dict = {}
    m._inject_pin((), {"model_settings": ms})
    provider = ms["extra_body"]["provider"]
    assert provider["order"] == ["DeepInfra"]
    assert provider["max_price"] == DEFAULT_MAX_PRICE_CHEAP
    assert provider["allow_fallbacks"] is True


def test_routing_never_forbids_fallbacks_by_default():
    """The 2026-07-29 board-wide outage: ``allow_fallbacks: False`` meant one
    dry upstream account 402'd every request while ~17 providers were healthy.
    A hard pin must not creep back in as a default on any tier."""
    for level in (1, 2):
        ms: dict = {}
        _model(level)._inject_pin((), {"model_settings": ms})
        routing = ms["extra_body"]["provider"]
        assert routing["allow_fallbacks"] is True
        assert "only" not in routing


def test_price_ceiling_is_configurable_per_bound():
    """Each bound overrides independently, so raising the completion ceiling
    does not silently reset the prompt one."""
    pytest.importorskip("pydantic_ai.providers.openrouter")
    from pydantic_ai.providers.openrouter import OpenRouterProvider as _Pyd

    from robotsix_llmio.openrouter._deepseek_model import (
        DEFAULT_MAX_PRICE_CAPABLE,
        OpenRouterDeepseekModel,
    )
    from robotsix_llmio.openrouter._deepseek_provider import (
        OpenRouterDeepseekProvider,
    )

    m = OpenRouterDeepseekModel("deepseek/deepseek-v4-pro", provider=_Pyd(api_key="x"))
    OpenRouterDeepseekProvider(api_key="x", max_price_completion=9.0)._post_build_model(
        m, 2
    )
    ms: dict = {}
    m._inject_pin((), {"model_settings": ms})
    assert ms["extra_body"]["provider"]["max_price"] == {
        "prompt": DEFAULT_MAX_PRICE_CAPABLE["prompt"],
        "completion": 9.0,
    }


def test_level2_default_ignores_cache_read_outliers():
    """DigitalOcean/CoreWeave cache-read rates are a multiple of DeepSeek's — a
    cost ``max_price`` cannot express — so the capable tier excludes them by
    default via ``provider.ignore``."""
    from robotsix_llmio.openrouter._deepseek_model import DEFAULT_IGNORE_CAPABLE

    ms: dict = {}
    _model(2)._inject_pin((), {"model_settings": ms})
    assert ms["extra_body"]["provider"]["ignore"] == list(DEFAULT_IGNORE_CAPABLE)


def test_level1_has_no_default_ignore():
    """The cheap tier has no cache-read outliers to exclude, so no ``ignore``
    key is injected by default."""
    ms: dict = {}
    _model(1)._inject_pin((), {"model_settings": ms})
    assert "ignore" not in ms["extra_body"]["provider"]


def test_ignore_providers_is_configurable():
    """``ignore_providers`` flows from the provider constructor (and hence
    ``provider_kwargs``) into ``provider.ignore``, overriding the default."""
    pytest.importorskip("pydantic_ai.providers.openrouter")
    from pydantic_ai.providers.openrouter import OpenRouterProvider as _Pyd

    from robotsix_llmio.openrouter._deepseek_model import OpenRouterDeepseekModel
    from robotsix_llmio.openrouter._deepseek_provider import (
        OpenRouterDeepseekProvider,
    )

    m = OpenRouterDeepseekModel("deepseek/deepseek-v4-pro", provider=_Pyd(api_key="x"))
    OpenRouterDeepseekProvider(
        api_key="x", ignore_providers=["SomeProvider"]
    )._post_build_model(m, 2)
    ms: dict = {}
    m._inject_pin((), {"model_settings": ms})
    assert ms["extra_body"]["provider"]["ignore"] == ["SomeProvider"]


def test_ignore_providers_empty_list_disables_default():
    """An explicit empty list opts the capable tier out of its default ignore."""
    pytest.importorskip("pydantic_ai.providers.openrouter")
    from pydantic_ai.providers.openrouter import OpenRouterProvider as _Pyd

    from robotsix_llmio.openrouter._deepseek_model import OpenRouterDeepseekModel
    from robotsix_llmio.openrouter._deepseek_provider import (
        OpenRouterDeepseekProvider,
    )

    m = OpenRouterDeepseekModel("deepseek/deepseek-v4-pro", provider=_Pyd(api_key="x"))
    OpenRouterDeepseekProvider(api_key="x", ignore_providers=[])._post_build_model(m, 2)
    ms: dict = {}
    m._inject_pin((), {"model_settings": ms})
    assert "ignore" not in ms["extra_body"]["provider"]


def test_preferred_provider_price_satisfies_capable_tier_ceiling():
    """The capable tier's ceiling must admit the preferred (DeepSeek)
    endpoint's own listed sticker price, so ``order`` and ``max_price``
    never contradict.

    Preferred price measured against the live endpoint list on
    2026-08-21: ``deepseek-v4-pro`` $0.660/$1.980.
    """
    from robotsix_llmio.openrouter._deepseek_model import (
        DEFAULT_MAX_PRICE_CAPABLE,
    )

    sticker = {"prompt": 0.660, "completion": 1.980}
    assert DEFAULT_MAX_PRICE_CAPABLE["prompt"] >= sticker["prompt"]
    assert DEFAULT_MAX_PRICE_CAPABLE["completion"] >= sticker["completion"]


def test_cheap_tier_ceiling_deliberately_excludes_repriced_deepseek():
    """The cheap tier's ceiling intentionally does NOT admit DeepSeek's
    own endpoint: DeepSeek repriced its flash serving to $0.22/$0.66
    (measured 2026-08-26) while several healthy endpoints serve the same
    snapshot under $0.10/$0.20 with cache-read rates matching DeepSeek's.
    Re-admitting DeepSeek would also re-admit the expensive fallback
    tail — see the module comment on ``DEFAULT_MAX_PRICE_CHEAP``.
    """
    from robotsix_llmio.openrouter._deepseek_model import (
        DEFAULT_MAX_PRICE_CHEAP,
    )

    deepseek_sticker = {"prompt": 0.220, "completion": 0.660}
    # Cheapest healthy endpoints measured 2026-08-26: OpenInference
    # $0.03/$0.075, Relace $0.06/$0.12, DeepInfra $0.08/$0.18, Makora
    # $0.09/$0.195 — the ceiling must keep admitting at least these.
    costliest_admitted = {"prompt": 0.09, "completion": 0.195}
    assert DEFAULT_MAX_PRICE_CHEAP["prompt"] < deepseek_sticker["prompt"]
    assert DEFAULT_MAX_PRICE_CHEAP["completion"] < deepseek_sticker["completion"]
    assert DEFAULT_MAX_PRICE_CHEAP["prompt"] >= costliest_admitted["prompt"]
    assert DEFAULT_MAX_PRICE_CHEAP["completion"] >= costliest_admitted["completion"]


def test_hard_pin_remains_available_as_an_explicit_opt_in():
    """Callers that genuinely need one provider can still say so — they just
    have to ask for it rather than get it by default."""
    pytest.importorskip("pydantic_ai.providers.openrouter")
    from pydantic_ai.providers.openrouter import OpenRouterProvider as _Pyd

    from robotsix_llmio.openrouter._deepseek_model import OpenRouterDeepseekModel
    from robotsix_llmio.openrouter._deepseek_provider import (
        OpenRouterDeepseekProvider,
    )

    m = OpenRouterDeepseekModel("deepseek/deepseek-v4-pro", provider=_Pyd(api_key="x"))
    OpenRouterDeepseekProvider(api_key="x", allow_fallbacks=False)._post_build_model(
        m, 2
    )
    ms: dict = {}
    m._inject_pin((), {"model_settings": ms})
    assert ms["extra_body"]["provider"]["allow_fallbacks"] is False


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


# --- non-DeepSeek model routing --------------------------------------------


def test_non_deepseek_model_has_no_deepseek_ceiling_or_order():
    """Non-DeepSeek models (e.g. openai/gpt-4o) must not carry a DeepSeek-
    derived max_price ceiling or order pin — only allow_fallbacks."""
    pytest.importorskip("pydantic_ai.providers.openrouter")
    from pydantic_ai.providers.openrouter import OpenRouterProvider as _Pyd

    from robotsix_llmio.openrouter._deepseek_model import OpenRouterDeepseekModel
    from robotsix_llmio.openrouter._deepseek_provider import (
        OpenRouterDeepseekProvider,
    )

    m = OpenRouterDeepseekModel("openai/gpt-4o", provider=_Pyd(api_key="x"))
    OpenRouterDeepseekProvider(api_key="x")._post_build_model(m, 2)
    ms: dict = {}
    m._inject_pin((), {"model_settings": ms})
    routing = ms["extra_body"]["provider"]
    assert routing["allow_fallbacks"] is True
    assert "order" not in routing
    assert "max_price" not in routing
    assert "ignore" not in routing


def test_non_deepseek_model_no_ceiling_at_level_1_too():
    """The guard applies regardless of level — level 1 also avoids the cheap-
    tier DeepSeek ceiling."""
    pytest.importorskip("pydantic_ai.providers.openrouter")
    from pydantic_ai.providers.openrouter import OpenRouterProvider as _Pyd

    from robotsix_llmio.openrouter._deepseek_model import OpenRouterDeepseekModel
    from robotsix_llmio.openrouter._deepseek_provider import (
        OpenRouterDeepseekProvider,
    )

    m = OpenRouterDeepseekModel("openai/gpt-4o", provider=_Pyd(api_key="x"))
    OpenRouterDeepseekProvider(api_key="x")._post_build_model(m, 1)
    ms: dict = {}
    m._inject_pin((), {"model_settings": ms})
    routing = ms["extra_body"]["provider"]
    assert "max_price" not in routing
    assert "order" not in routing


def test_non_deepseek_model_honours_explicit_routing_kwargs():
    """Explicit routing knobs (how a tier's ``provider_kwargs`` arrive) apply
    to non-DeepSeek models too; only the reasoning policy stays DeepSeek-only.

    Regression: until 2026-08-28 these were silently dropped for ``xiaomi/``
    models, so OpenRouter free-routed level 2 onto providers whose cache-read
    rate is 20-45x Xiaomi's."""
    pytest.importorskip("pydantic_ai.providers.openrouter")
    from pydantic_ai.providers.openrouter import OpenRouterProvider as _Pyd

    from robotsix_llmio.openrouter._deepseek_model import OpenRouterDeepseekModel
    from robotsix_llmio.openrouter._deepseek_provider import (
        OpenRouterDeepseekProvider,
    )

    m = OpenRouterDeepseekModel("xiaomi/mimo-v2.5-pro", provider=_Pyd(api_key="x"))
    OpenRouterDeepseekProvider(
        api_key="x",
        preferred_provider="Xiaomi",
        max_price_prompt=0.55,
        max_price_completion=1.10,
        ignore_providers=["DigitalOcean", "DeepInfra"],
    )._post_build_model(m, 2)
    ms: dict = {}
    m._inject_pin((), {"model_settings": ms})
    routing = ms["extra_body"]["provider"]
    assert routing == {
        "allow_fallbacks": True,
        "order": ["Xiaomi"],
        "max_price": {"prompt": 0.55, "completion": 1.10},
        "ignore": ["DigitalOcean", "DeepInfra"],
    }
    # Reasoning policy remains DeepSeek-specific.
    assert "reasoning" not in ms["extra_body"]


def test_non_deepseek_model_partial_ceiling_sends_only_given_bounds():
    """A single explicit bound yields a one-key ceiling — never a DeepSeek
    default for the missing half."""
    pytest.importorskip("pydantic_ai.providers.openrouter")
    from pydantic_ai.providers.openrouter import OpenRouterProvider as _Pyd

    from robotsix_llmio.openrouter._deepseek_model import OpenRouterDeepseekModel
    from robotsix_llmio.openrouter._deepseek_provider import (
        OpenRouterDeepseekProvider,
    )

    m = OpenRouterDeepseekModel("xiaomi/mimo-v2.5-pro", provider=_Pyd(api_key="x"))
    OpenRouterDeepseekProvider(
        api_key="x", max_price_completion=1.10
    )._post_build_model(m, 2)
    ms: dict = {}
    m._inject_pin((), {"model_settings": ms})
    routing = ms["extra_body"]["provider"]
    assert routing["max_price"] == {"completion": 1.10}
    assert "order" not in routing
    assert "ignore" not in routing


def test_deepseek_model_default_preference_still_deepseek():
    """The per-model sentinel resolves to DeepSeek for ``deepseek/`` models,
    so omitting ``preferred_provider`` keeps the historical pin."""
    pytest.importorskip("pydantic_ai.providers.openrouter")
    from pydantic_ai.providers.openrouter import OpenRouterProvider as _Pyd

    from robotsix_llmio.openrouter._deepseek_model import OpenRouterDeepseekModel
    from robotsix_llmio.openrouter._deepseek_provider import (
        OpenRouterDeepseekProvider,
    )

    m = OpenRouterDeepseekModel("deepseek/deepseek-v4-pro", provider=_Pyd(api_key="x"))
    OpenRouterDeepseekProvider(api_key="x")._post_build_model(m, 2)
    ms: dict = {}
    m._inject_pin((), {"model_settings": ms})
    assert ms["extra_body"]["provider"]["order"] == ["DeepSeek"]


def test_level3_default_tier_routing_reaches_the_request():
    """End-to-end through the factory: the baked level-3 ``provider_kwargs``
    (Xiaomi preferred, ceiling, ignore list) must land in the request body.
    This is the path mill and chat use — the one that was silently dropping
    them."""
    pytest.importorskip("pydantic_ai.providers.openrouter")

    from robotsix_llmio.config.tier import LEVEL3_DEFAULT
    from robotsix_llmio.core.factory import get_provider_for_level

    provider = get_provider_for_level(3, api_key="x")
    model, http_client = provider.new_model(model=LEVEL3_DEFAULT.model_name, level=3)
    try:
        ms: dict = {}
        model._inject_pin((), {"model_settings": ms})
        routing = ms["extra_body"]["provider"]
        expected = LEVEL3_DEFAULT.provider_kwargs
        assert routing["order"] == [expected["preferred_provider"]]
        assert routing["max_price"] == {
            "prompt": expected["max_price_prompt"],
            "completion": expected["max_price_completion"],
        }
        assert routing["ignore"] == expected["ignore_providers"]
        assert routing["allow_fallbacks"] is True
    finally:
        import asyncio

        asyncio.run(http_client.aclose())


# --- response key constants ------------------------------------------------


def test_deepseek_response_key_constants_pin_wire_values():
    """The extracted constants must keep their exact DeepSeek wire strings."""
    from robotsix_llmio.openrouter import _deepseek_model as ds_model

    assert ds_model._REASONING_KEY == "reasoning"
    assert ds_model._REASONING_CONTENT_KEY == "reasoning_content"
    assert ds_model._TOOL_CALLS_KEY == "tool_calls"

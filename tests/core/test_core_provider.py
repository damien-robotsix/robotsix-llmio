"""Core provider base class — Tier enum, _is_transient default, and the
``build_agent`` / ``call_with_retry`` wiring that every concrete provider
inherits."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import httpx
import pytest

from robotsix_llmio.config.tier import TierConfig, TierLevelConfig
from robotsix_llmio.core import provider as provider_module
from robotsix_llmio.core import retry as retry_module
from robotsix_llmio.core.provider import LLMProvider


class _HTTPErr(Exception):
    """Cheap stand-in for a ``ModelHTTPError`` — only the ``status_code`` attr
    matters for transient classification."""

    def __init__(self, status_code: int) -> None:
        self.status_code = status_code


class _MockProvider(LLMProvider):
    """Bare-minimum concrete provider: implements ``new_model`` only and
    records what it was asked for."""

    def __init__(self, model: Any = None, http_client: Any = None) -> None:
        self.model_obj = model if model is not None else object()
        self.http_client_obj = http_client if http_client is not None else object()
        self.new_model_calls: list[dict[str, Any]] = []

    def new_model(
        self,
        *,
        model: str | None = None,
        level: int = 0,
    ) -> tuple[Any, Any]:
        self.new_model_calls.append({"model": model, "level": level})
        return self.model_obj, self.http_client_obj


# --- shared fixture: mock _build_agent -------------------------------------


@pytest.fixture
def mock_build_agent(monkeypatch):
    def _mock(model=None, http_client=None, **kwargs):
        return SimpleNamespace()

    monkeypatch.setattr(provider_module, "_build_agent", _mock)
    return _mock


# ---------------------------------------------------------------------------


def test_new_model_defaults():
    """Without arguments, ``new_model`` receives ``model=None, level=0``
    (the sentinel defaults)."""
    p = _MockProvider()
    p.new_model()
    assert p.new_model_calls == [{"model": None, "level": 0}]


# --- _is_transient default delegates to retry.is_transient ------------------


def test_is_transient_default_delegates_to_retry_is_transient():
    p = _MockProvider()
    # 429/5xx → transient; 4xx/other → not transient. These are the
    # behaviours owned by ``retry.is_transient``; the base provider must
    # forward verbatim.
    assert p._is_transient(_HTTPErr(503)) is True
    assert p._is_transient(_HTTPErr(429)) is True
    assert p._is_transient(_HTTPErr(400)) is False
    assert p._is_transient(_HTTPErr(404)) is False
    assert p._is_transient(ValueError("boom")) is False


def test_is_transient_default_recognises_httpx_timeout():
    p = _MockProvider()
    assert p._is_transient(httpx.ReadTimeout("slow")) is True
    assert p._is_transient(httpx.ConnectError("refused")) is True


def test_is_transient_default_calls_retry_module(monkeypatch):
    """The default implementation must funnel through ``retry.is_transient``
    so provider layers can widen by overriding."""
    seen: list[BaseException] = []

    def fake(exc: BaseException) -> bool:
        seen.append(exc)
        return True

    monkeypatch.setattr(retry_module, "is_transient", fake)
    p = _MockProvider()
    err = ValueError("probe")
    assert p._is_transient(err) is True
    assert seen == [err]


def test_is_transient_override_is_used_by_call_with_retry():
    """A subclass that widens ``_is_transient`` causes ``call_with_retry`` to
    retry on the wider error set."""

    class _ValueErrProvider(_MockProvider):
        def _is_transient(self, exc: BaseException) -> bool:
            return isinstance(exc, ValueError)

    p = _ValueErrProvider()
    calls = {"n": 0}

    def fn():
        calls["n"] += 1
        if calls["n"] < 2:
            raise ValueError("provider-specific transient")
        return "ok"

    out = p.call_with_retry(fn, sleep=lambda _d: None)
    assert out == "ok"
    assert calls["n"] == 2


# --- build_agent wiring (tier_config default path) -------------------------


def test_build_agent_calls_new_model_with_model_name(monkeypatch):
    """When ``tier_config`` is not provided, ``build_agent`` constructs a
    default ``TierConfig`` from baked defaults and calls
    ``new_model(model=tlc.model_name)``."""
    p = _MockProvider()
    captured: dict[str, Any] = {}

    def fake_build_agent(model, http_client, **kwargs):
        captured["model"] = model
        captured["http_client"] = http_client
        captured.update(kwargs)
        return SimpleNamespace(_agent=model)

    monkeypatch.setattr(provider_module, "_build_agent", fake_build_agent)
    p.build_agent(system_prompt="sys")
    # Default level=1 → LEVEL1_DEFAULT.model = "deepseek/deepseek-v4-flash-latest"
    assert p.new_model_calls == [
        {"model": "deepseek/deepseek-v4-flash-latest", "level": 1},
    ]
    assert captured["model"] is p.model_obj
    assert captured["http_client"] is p.http_client_obj
    assert captured["system_prompt"] == "sys"


@pytest.mark.parametrize(
    ("level", "expected_model"),
    [
        (1, "deepseek/deepseek-v4-flash-latest"),
        (2, "xiaomi/mimo-v2.5-pro"),
        (3, "opus"),
        (4, "claude-fable-5"),
    ],
)
def test_build_agent_level_uses_default(mock_build_agent, level, expected_model):
    p = _MockProvider()
    p.build_agent(level=level, system_prompt="sys")
    assert p.new_model_calls == [{"model": expected_model, "level": level}]


def test_build_agent_level_out_of_range_raises(mock_build_agent):
    p = _MockProvider()
    with pytest.raises(ValueError, match=r"`level` must be 1, 2, 3, or 4, got 0"):
        p.build_agent(level=0, system_prompt="sys")
    with pytest.raises(ValueError, match=r"`level` must be 1, 2, 3, or 4, got 5"):
        p.build_agent(level=5, system_prompt="sys")


def test_build_agent_threads_kwargs(monkeypatch):
    p = _MockProvider()
    captured: dict[str, Any] = {}

    def fake_build_agent(model, http_client, **kwargs):
        captured["model"] = model
        captured["http_client"] = http_client
        captured.update(kwargs)
        return SimpleNamespace()

    monkeypatch.setattr(provider_module, "_build_agent", fake_build_agent)

    def _tool() -> None:  # pragma: no cover — never invoked
        return None

    tools = [_tool]
    p.build_agent(
        system_prompt="sys",
        tools=tools,
        output_type=dict,
        name="my-agent",
        retries=7,
    )
    assert captured["system_prompt"] == "sys"
    assert captured["tools"] is tools
    assert captured["output_type"] is dict
    assert captured["name"] == "my-agent"
    assert captured["retries"] == 7


def test_build_agent_returns_underlying_handle(monkeypatch):
    """``build_agent`` returns exactly what the inner ``_build_agent`` returns."""
    sentinel = SimpleNamespace(marker="handle")

    def fake_build_agent(*_args, **_kwargs):
        return sentinel

    monkeypatch.setattr(provider_module, "_build_agent", fake_build_agent)
    p = _MockProvider()
    assert p.build_agent(system_prompt="sys") is sentinel


# --- build_agent model override --------------------------------------------


def test_build_agent_model_override_bypasses_tier_config(mock_build_agent):
    """When ``model`` is provided, ``new_model`` is called with the explicit
    model name and *tier_config* is never consulted (no ValueError for
    out-of-range levels either — level is still passed)."""
    p = _MockProvider()
    p.build_agent(model="my-custom-model", level=1, system_prompt="sys")
    assert p.new_model_calls == [{"model": "my-custom-model", "level": 1}]


def test_build_agent_model_override_with_level(mock_build_agent):
    """``level`` is still forwarded alongside the explicit model."""
    p = _MockProvider()
    p.build_agent(model="custom", level=3, system_prompt="sys")
    assert p.new_model_calls == [{"model": "custom", "level": 3}]


def test_build_agent_model_override_wins_over_tier_config(mock_build_agent):
    """Even when ``tier_config`` is provided, the explicit ``model`` takes
    precedence."""
    cfg = TierConfig(
        level1=TierLevelConfig(model="claudeSDK-opus"),
    )
    p = _MockProvider()
    p.build_agent(
        model="overridden-model", level=1, tier_config=cfg, system_prompt="sys"
    )
    assert p.new_model_calls == [{"model": "overridden-model", "level": 1}]


def test_build_agent_model_none_still_resolves_from_tier_config(mock_build_agent):
    """When ``model=None`` (the default), tier_config resolution works as before."""
    p = _MockProvider()
    p.build_agent(level=1, system_prompt="sys")
    # LEVEL1_DEFAULT.model = "deepseek/deepseek-v4-flash-latest"
    assert p.new_model_calls == [
        {"model": "deepseek/deepseek-v4-flash-latest", "level": 1},
    ]


# --- build_agent primary path (tier_config provided) ------------------------


@pytest.mark.parametrize(
    ("level", "tier_config_kwargs", "expected_model"),
    [
        (
            1,
            {"level1": TierLevelConfig(model="claudeSDK-opus")},
            "opus",
        ),
        (
            2,
            {
                "level1": TierLevelConfig(model="claudeSDK-opus"),
                "level2": TierLevelConfig(model="claudeSDK-haiku"),
            },
            "haiku",
        ),
        (
            3,
            {
                "level1": TierLevelConfig(model="claudeSDK-opus"),
                "level3": TierLevelConfig(model="claudeSDK-sonnet"),
            },
            "sonnet",
        ),
    ],
)
def test_build_agent_with_tier_config(
    mock_build_agent, level, tier_config_kwargs, expected_model
):
    """Primary path: ``build_agent(level=N, tier_config=cfg)`` calls
    ``new_model(model=cfg.levelN.model_name)``."""
    cfg = TierConfig(**tier_config_kwargs)
    p = _MockProvider()
    p.build_agent(level=level, tier_config=cfg, system_prompt="sys")
    assert p.new_model_calls == [{"model": expected_model, "level": level}]


# --- call_with_retry delegation --------------------------------------------


def test_call_with_retry_default_predicate_does_not_retry_valueerror():
    """With the default ``_is_transient``, a plain ``ValueError`` is fatal."""
    p = _MockProvider()
    calls = {"n": 0}

    def fn():
        calls["n"] += 1
        raise ValueError("not transient by default")

    with pytest.raises(ValueError):
        p.call_with_retry(fn, sleep=lambda _d: None)
    assert calls["n"] == 1


def test_call_with_retry_passes_through_to_retry_module(monkeypatch):
    """The base ``call_with_retry`` must hand its arguments to
    ``retry.call_with_retry`` and pin ``is_transient_fn`` to the provider's
    own ``_is_transient``."""
    p = _MockProvider()
    captured: dict[str, Any] = {}

    def fake_call_with_retry(fn, **kwargs):
        captured["fn"] = fn
        captured.update(kwargs)
        return "ok"

    monkeypatch.setattr(retry_module, "call_with_retry", fake_call_with_retry)

    def target():  # pragma: no cover — never invoked
        return "x"

    def sleep(_d: float) -> None:  # pragma: no cover — never invoked
        return None

    def fallback():  # pragma: no cover — never invoked
        return "fb"

    out = p.call_with_retry(target, what="probe", sleep=sleep, fallback_fn=fallback)
    assert out == "ok"
    assert captured["fn"] is target
    assert captured["what"] == "probe"
    assert captured["sleep"] is sleep
    assert captured["fallback_fn"] is fallback
    # The predicate must be the provider's own bound ``_is_transient`` so
    # subclass overrides take effect.
    assert captured["is_transient_fn"] == p._is_transient


def test_call_with_retry_retries_on_transient_5xx():
    """End-to-end: the provider's default predicate accepts 5xx, so the base
    ``call_with_retry`` recovers transient HTTP errors transparently."""
    p = _MockProvider()
    calls = {"n": 0}

    def fn():
        calls["n"] += 1
        if calls["n"] < 3:
            raise _HTTPErr(503)
        return "ok"

    out = p.call_with_retry(fn, sleep=lambda _d: None)
    assert out == "ok"
    assert calls["n"] == 3


# ---- _resolve_output_type unit tests ------------------------------------


class _ExampleModel:
    """Stand-in for a pydantic BaseModel subclass (lazy import avoided)."""

    pass


def test_resolve_output_type_level_1_raw_type_passthrough():
    """At level 1, a raw type is passed through unchanged."""
    from robotsix_llmio.core.provider import _resolve_output_type

    assert _resolve_output_type(_ExampleModel, 1) is _ExampleModel


def test_resolve_output_type_level_2_str_passthrough():
    """str is always passed through, even at level 2."""
    from robotsix_llmio.core.provider import _resolve_output_type

    assert _resolve_output_type(str, 2) is str


def test_resolve_output_type_level_2_raw_type_wrapped():
    """At level 2, a raw type is wrapped in PromptedOutput."""
    from pydantic_ai import PromptedOutput

    from robotsix_llmio.core.provider import _resolve_output_type

    result = _resolve_output_type(_ExampleModel, 2)
    assert isinstance(result, PromptedOutput)
    assert result.outputs is _ExampleModel


def test_resolve_output_type_level_2_prompted_output_unchanged():
    """An explicit PromptedOutput is never double-wrapped."""
    from pydantic_ai import PromptedOutput

    from robotsix_llmio.core.provider import _resolve_output_type

    po = PromptedOutput(_ExampleModel)
    result = _resolve_output_type(po, 2)
    assert result is po


def test_resolve_output_type_level_2_tool_output_unchanged():
    """An explicit ToolOutput is never double-wrapped."""
    from pydantic_ai import ToolOutput

    from robotsix_llmio.core.provider import _resolve_output_type

    to = ToolOutput(_ExampleModel)
    result = _resolve_output_type(to, 2)
    assert result is to


def test_resolve_output_type_level_2_native_output_unchanged():
    """An explicit NativeOutput is never double-wrapped."""
    from pydantic_ai import NativeOutput

    from robotsix_llmio.core.provider import _resolve_output_type

    no = NativeOutput(_ExampleModel)
    result = _resolve_output_type(no, 2)
    assert result is no


def test_resolve_output_type_level_2_list_of_raw_types_wrapped():
    """At level 2, a list of raw types is wrapped in PromptedOutput."""
    from pydantic_ai import PromptedOutput

    from robotsix_llmio.core.provider import _resolve_output_type

    result = _resolve_output_type([_ExampleModel, str], 2)
    assert isinstance(result, PromptedOutput)
    assert result.outputs == [_ExampleModel, str]


def test_resolve_output_type_level_2_list_with_marker_unchanged():
    """A list containing an explicit marker is left unchanged."""
    from pydantic_ai import ToolOutput

    from robotsix_llmio.core.provider import _resolve_output_type

    to = ToolOutput(_ExampleModel)
    mixed = [_ExampleModel, to]
    assert _resolve_output_type(mixed, 2) is mixed


def test_resolve_output_type_level_0_unchanged():
    """At level 0 (sentinel), output_type is unchanged."""
    from robotsix_llmio.core.provider import _resolve_output_type

    assert _resolve_output_type(_ExampleModel, 0) is _ExampleModel


def test_resolve_output_type_level_2_with_pydantic_basemodel():
    """At level 2, a real pydantic BaseModel subclass is wrapped."""
    from pydantic import BaseModel
    from pydantic_ai import PromptedOutput

    from robotsix_llmio.core.provider import _resolve_output_type

    class _RealModel(BaseModel):
        x: int

    result = _resolve_output_type(_RealModel, 2)
    assert isinstance(result, PromptedOutput)
    assert result.outputs is _RealModel


# ---- _resolve_output_type integrated via build_agent ----------------------


def test_build_agent_level_2_raw_type_passed_as_prompted_output(monkeypatch):
    """At level=2, ``build_agent`` wraps a raw pydantic type in PromptedOutput
    before passing it to ``_build_agent``."""
    from pydantic import BaseModel
    from pydantic_ai import PromptedOutput

    from robotsix_llmio.core import provider as provider_module

    class _Foo(BaseModel):
        bar: str

    p = _MockProvider()
    captured: dict[str, Any] = {}

    def fake_build_agent(model, http_client, **kwargs):
        captured["model"] = model
        captured["http_client"] = http_client
        captured.update(kwargs)
        return SimpleNamespace()

    monkeypatch.setattr(provider_module, "_build_agent", fake_build_agent)
    p.build_agent(level=2, system_prompt="sys", output_type=_Foo)
    passed = captured["output_type"]
    assert isinstance(passed, PromptedOutput)
    assert passed.outputs is _Foo


def test_build_agent_level_2_explicit_tool_output_unchanged(monkeypatch):
    """At level=2 with an explicit ToolOutput, the exact instance is
    forwarded to ``_build_agent`` (no double-wrap)."""
    from pydantic import BaseModel
    from pydantic_ai import ToolOutput

    from robotsix_llmio.core import provider as provider_module

    class _Foo(BaseModel):
        bar: str

    p = _MockProvider()
    captured: dict[str, Any] = {}

    def fake_build_agent(model, http_client, **kwargs):
        captured.update(kwargs)
        return SimpleNamespace()

    monkeypatch.setattr(provider_module, "_build_agent", fake_build_agent)
    to = ToolOutput(_Foo)
    p.build_agent(level=2, system_prompt="sys", output_type=to)
    assert captured["output_type"] is to


def test_build_agent_level_1_raw_type_not_wrapped(monkeypatch):
    """At level=1, a raw type is NOT wrapped."""
    from pydantic import BaseModel

    from robotsix_llmio.core import provider as provider_module

    class _Foo(BaseModel):
        bar: str

    p = _MockProvider()
    captured: dict[str, Any] = {}

    def fake_build_agent(model, http_client, **kwargs):
        captured.update(kwargs)
        return SimpleNamespace()

    monkeypatch.setattr(provider_module, "_build_agent", fake_build_agent)
    p.build_agent(level=1, system_prompt="sys", output_type=_Foo)
    assert captured["output_type"] is _Foo


def test_build_agent_model_override_level_2_wraps(monkeypatch):
    """The model-override branch also applies the wrap at level 2."""
    from pydantic import BaseModel
    from pydantic_ai import PromptedOutput

    from robotsix_llmio.core import provider as provider_module

    class _Foo(BaseModel):
        bar: str

    p = _MockProvider()
    captured: dict[str, Any] = {}

    def fake_build_agent(model, http_client, **kwargs):
        captured.update(kwargs)
        return SimpleNamespace()

    monkeypatch.setattr(provider_module, "_build_agent", fake_build_agent)
    p.build_agent(model="custom", level=2, system_prompt="sys", output_type=_Foo)
    passed = captured["output_type"]
    assert isinstance(passed, PromptedOutput)
    assert passed.outputs is _Foo


# ---- _is_prompted_output unit tests --------------------------------------


def test_is_prompted_output_with_prompted_output_instance():
    from pydantic_ai import PromptedOutput

    from robotsix_llmio.core.provider import _is_prompted_output

    assert _is_prompted_output(PromptedOutput(str)) is True


def test_is_prompted_output_with_tool_output_is_false():
    from pydantic_ai import ToolOutput

    from robotsix_llmio.core.provider import _is_prompted_output

    assert _is_prompted_output(ToolOutput(str)) is False


def test_is_prompted_output_with_native_output_is_false():
    from pydantic_ai import NativeOutput

    from robotsix_llmio.core.provider import _is_prompted_output

    assert _is_prompted_output(NativeOutput(str)) is False


def test_is_prompted_output_with_str_is_false():
    from robotsix_llmio.core.provider import _is_prompted_output

    assert _is_prompted_output(str) is False


def test_is_prompted_output_with_raw_type_is_false():
    from robotsix_llmio.core.provider import _is_prompted_output

    assert _is_prompted_output(dict) is False


def test_is_prompted_output_with_list_containing_prompted():
    from pydantic_ai import PromptedOutput

    from robotsix_llmio.core.provider import _is_prompted_output

    assert _is_prompted_output([PromptedOutput(str), dict]) is True


def test_is_prompted_output_with_empty_list_is_false():
    from robotsix_llmio.core.provider import _is_prompted_output

    assert _is_prompted_output([]) is False


# ---- anti-DSML instruction injection -------------------------------------


def test_build_agent_injects_anti_dsml_for_prompted_output(monkeypatch):
    """When the resolved output type is PromptedOutput, the system prompt
    is augmented with an anti-DSML instruction."""
    from pydantic_ai import PromptedOutput

    from robotsix_llmio.core import provider as provider_module
    from robotsix_llmio.core.provider import _ANTI_DSML_INSTRUCTION

    p = _MockProvider()
    captured: dict[str, Any] = {}

    def fake_build_agent(model, http_client, **kwargs):
        captured.update(kwargs)
        return SimpleNamespace()

    monkeypatch.setattr(provider_module, "_build_agent", fake_build_agent)
    p.build_agent(level=1, system_prompt="Be helpful.", output_type=PromptedOutput(str))
    assert captured["system_prompt"] == "Be helpful." + _ANTI_DSML_INSTRUCTION


def test_build_agent_no_anti_dsml_for_str_output(monkeypatch):
    """When output_type is str, the system prompt is NOT augmented."""
    from robotsix_llmio.core import provider as provider_module

    p = _MockProvider()
    captured: dict[str, Any] = {}

    def fake_build_agent(model, http_client, **kwargs):
        captured.update(kwargs)
        return SimpleNamespace()

    monkeypatch.setattr(provider_module, "_build_agent", fake_build_agent)
    p.build_agent(level=1, system_prompt="Be helpful.", output_type=str)
    assert captured["system_prompt"] == "Be helpful."


def test_build_agent_no_anti_dsml_for_tool_output(monkeypatch):
    """When output_type is ToolOutput, the system prompt is NOT augmented."""
    from pydantic_ai import ToolOutput

    from robotsix_llmio.core import provider as provider_module

    p = _MockProvider()
    captured: dict[str, Any] = {}

    def fake_build_agent(model, http_client, **kwargs):
        captured.update(kwargs)
        return SimpleNamespace()

    monkeypatch.setattr(provider_module, "_build_agent", fake_build_agent)
    p.build_agent(level=1, system_prompt="Be helpful.", output_type=ToolOutput(str))
    assert captured["system_prompt"] == "Be helpful."


def test_build_agent_injects_anti_dsml_for_level2_raw_type(monkeypatch):
    """At level=2 a raw pydantic type is wrapped in PromptedOutput, so the
    anti-DSML instruction is injected."""
    from pydantic import BaseModel

    from robotsix_llmio.core import provider as provider_module
    from robotsix_llmio.core.provider import _ANTI_DSML_INSTRUCTION

    class _Foo(BaseModel):
        bar: str

    p = _MockProvider()
    captured: dict[str, Any] = {}

    def fake_build_agent(model, http_client, **kwargs):
        captured.update(kwargs)
        return SimpleNamespace()

    monkeypatch.setattr(provider_module, "_build_agent", fake_build_agent)
    p.build_agent(level=2, system_prompt="sys", output_type=_Foo)
    assert captured["system_prompt"] == "sys" + _ANTI_DSML_INSTRUCTION

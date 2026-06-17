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
from robotsix_llmio.core.provider import LLMProvider, Tier


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


# --- Tier enum --------------------------------------------------------------


def test_tier_values():
    assert Tier.DEFAULT.value == "default"
    assert Tier.CHEAP.value == "cheap"


def test_tier_is_str_enum():
    # ``str, Enum`` mixin: instances are both str and Enum so they can be
    # compared with plain string literals.
    assert isinstance(Tier.DEFAULT, str)
    assert isinstance(Tier.CHEAP, str)
    assert Tier.DEFAULT == "default"
    assert Tier.CHEAP == "cheap"


def test_tier_members():
    assert {t.name for t in Tier} == {"DEFAULT", "CHEAP"}


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
    ``new_model(model=tlc.model)``."""
    p = _MockProvider()
    captured: dict[str, Any] = {}

    def fake_build_agent(model, http_client, **kwargs):
        captured["model"] = model
        captured["http_client"] = http_client
        captured.update(kwargs)
        return SimpleNamespace(_agent=model)

    monkeypatch.setattr(provider_module, "_build_agent", fake_build_agent)
    p.build_agent(system_prompt="sys")
    # Default level=1 → LEVEL1_DEFAULT.model = "deepseek/deepseek-v4-flash"
    assert p.new_model_calls == [{"model": "deepseek/deepseek-v4-flash", "level": 1}]
    assert captured["model"] is p.model_obj
    assert captured["http_client"] is p.http_client_obj
    assert captured["system_prompt"] == "sys"


def test_build_agent_default_level_is_1(monkeypatch):
    """With no ``level`` argument, ``build_agent`` defaults to ``level=1``."""
    p = _MockProvider()

    def fake_build_agent(*_args, **_kwargs):
        return SimpleNamespace()

    monkeypatch.setattr(provider_module, "_build_agent", fake_build_agent)
    p.build_agent(system_prompt="sys")
    # LEVEL1_DEFAULT.model is "deepseek/deepseek-v4-flash" at level=1
    assert p.new_model_calls == [{"model": "deepseek/deepseek-v4-flash", "level": 1}]


def test_build_agent_level_2_uses_level2_default(monkeypatch):
    p = _MockProvider()

    def fake_build_agent(*_args, **_kwargs):
        return SimpleNamespace()

    monkeypatch.setattr(provider_module, "_build_agent", fake_build_agent)
    p.build_agent(level=2, system_prompt="sys")
    # LEVEL2_DEFAULT.model is "deepseek/deepseek-v4-pro"
    assert p.new_model_calls == [{"model": "deepseek/deepseek-v4-pro", "level": 2}]


def test_build_agent_level_3_uses_level3_default(monkeypatch):
    p = _MockProvider()

    def fake_build_agent(*_args, **_kwargs):
        return SimpleNamespace()

    monkeypatch.setattr(provider_module, "_build_agent", fake_build_agent)
    p.build_agent(level=3, system_prompt="sys")
    # LEVEL3_DEFAULT.model is "opus"
    assert p.new_model_calls == [{"model": "opus", "level": 3}]


def test_build_agent_level_out_of_range_raises(monkeypatch):
    p = _MockProvider()

    def fake_build_agent(*_args, **_kwargs):
        return SimpleNamespace()

    monkeypatch.setattr(provider_module, "_build_agent", fake_build_agent)
    with pytest.raises(ValueError, match=r"`level` must be 1, 2, or 3, got 0"):
        p.build_agent(level=0, system_prompt="sys")
    with pytest.raises(ValueError, match=r"`level` must be 1, 2, or 3, got 4"):
        p.build_agent(level=4, system_prompt="sys")


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


def test_build_agent_model_override_bypasses_tier_config(monkeypatch):
    """When ``model`` is provided, ``new_model`` is called with the explicit
    model name and *tier_config* is never consulted (no ValueError for
    out-of-range levels either — level is still passed)."""
    p = _MockProvider()

    def fake_build_agent(*_args, **_kwargs):
        return SimpleNamespace()

    monkeypatch.setattr(provider_module, "_build_agent", fake_build_agent)
    p.build_agent(model="my-custom-model", level=1, system_prompt="sys")
    assert p.new_model_calls == [{"model": "my-custom-model", "level": 1}]


def test_build_agent_model_override_with_level(monkeypatch):
    """``level`` is still forwarded alongside the explicit model."""
    p = _MockProvider()

    def fake_build_agent(*_args, **_kwargs):
        return SimpleNamespace()

    monkeypatch.setattr(provider_module, "_build_agent", fake_build_agent)
    p.build_agent(model="custom", level=3, system_prompt="sys")
    assert p.new_model_calls == [{"model": "custom", "level": 3}]


def test_build_agent_model_override_wins_over_tier_config(monkeypatch):
    """Even when ``tier_config`` is provided, the explicit ``model`` takes
    precedence."""
    cfg = TierConfig(
        level1=TierLevelConfig(transport="claude-sdk", model="opus"),
    )
    p = _MockProvider()

    def fake_build_agent(*_args, **_kwargs):
        return SimpleNamespace()

    monkeypatch.setattr(provider_module, "_build_agent", fake_build_agent)
    p.build_agent(
        model="overridden-model", level=1, tier_config=cfg, system_prompt="sys"
    )
    assert p.new_model_calls == [{"model": "overridden-model", "level": 1}]


def test_build_agent_model_none_still_resolves_from_tier_config(monkeypatch):
    """When ``model=None`` (the default), tier_config resolution works as before."""
    p = _MockProvider()

    def fake_build_agent(*_args, **_kwargs):
        return SimpleNamespace()

    monkeypatch.setattr(provider_module, "_build_agent", fake_build_agent)
    p.build_agent(level=1, system_prompt="sys")
    # LEVEL1_DEFAULT.model = "deepseek/deepseek-v4-flash"
    assert p.new_model_calls == [{"model": "deepseek/deepseek-v4-flash", "level": 1}]


# --- build_agent primary path (tier_config provided) ------------------------


def test_build_agent_with_tier_config_level_1(monkeypatch):
    """Primary path: ``build_agent(level=1, tier_config=cfg)`` calls
    ``new_model(model=cfg.level1.model)``."""
    cfg = TierConfig(
        level1=TierLevelConfig(transport="claude-sdk", model="opus"),
    )
    p = _MockProvider()

    def fake_build_agent(*_args, **_kwargs):
        return SimpleNamespace()

    monkeypatch.setattr(provider_module, "_build_agent", fake_build_agent)
    p.build_agent(level=1, tier_config=cfg, system_prompt="sys")
    assert p.new_model_calls == [{"model": "opus", "level": 1}]


def test_build_agent_with_tier_config_level_2(monkeypatch):
    """Primary path: ``build_agent(level=2, tier_config=cfg)`` calls
    ``new_model(model=cfg.level2.model)``."""
    cfg = TierConfig(
        level1=TierLevelConfig(transport="claude-sdk", model="opus"),
        level2=TierLevelConfig(transport="claude-sdk", model="haiku"),
    )
    p = _MockProvider()

    def fake_build_agent(*_args, **_kwargs):
        return SimpleNamespace()

    monkeypatch.setattr(provider_module, "_build_agent", fake_build_agent)
    p.build_agent(level=2, tier_config=cfg, system_prompt="sys")
    assert p.new_model_calls == [{"model": "haiku", "level": 2}]


def test_build_agent_with_tier_config_level_3(monkeypatch):
    """Primary path: ``build_agent(level=3, tier_config=cfg)`` calls
    ``new_model(model=cfg.level3.model)``."""
    cfg = TierConfig(
        level1=TierLevelConfig(transport="claude-sdk", model="opus"),
        level3=TierLevelConfig(transport="claude-sdk", model="sonnet"),
    )
    p = _MockProvider()

    def fake_build_agent(*_args, **_kwargs):
        return SimpleNamespace()

    monkeypatch.setattr(provider_module, "_build_agent", fake_build_agent)
    p.build_agent(level=3, tier_config=cfg, system_prompt="sys")
    assert p.new_model_calls == [{"model": "sonnet", "level": 3}]


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

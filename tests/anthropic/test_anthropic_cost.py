"""Offline tests for client-side Anthropic cost computation."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from robotsix_llmio.anthropic.pricing import (
    AnthropicPricing,
    ModelPrices,
    compute_cost,
    record_anthropic_cost,
)


def _usage(**kw: int) -> SimpleNamespace:
    base = {
        "input_tokens": 0,
        "output_tokens": 0,
        "cache_read_tokens": 0,
        "cache_write_tokens": 0,
    }
    base.update(kw)
    return SimpleNamespace(**base)


def test_compute_cost_splits_cache_read_and_write() -> None:
    prices = ModelPrices(input=4.0, output=20.0, cache_read=0.2, cache_write=5.0)
    # pydantic-ai input_tokens is the TOTAL input (uncached + cache read/write).
    usage = _usage(
        input_tokens=1300,
        output_tokens=50,
        cache_read_tokens=1000,
        cache_write_tokens=200,
    )
    expected = (100 * 4.0 + 1000 * 0.2 + 200 * 5.0 + 50 * 20.0) / 1_000_000
    assert compute_cost(usage, prices) == pytest.approx(expected)


def test_compute_cost_none_inputs() -> None:
    assert (
        compute_cost(None, ModelPrices(input=1, output=1, cache_read=1, cache_write=1))
        is None
    )
    assert compute_cost(_usage(), None) is None


def test_default_prices_cover_models_in_use() -> None:
    pricing = AnthropicPricing()
    for model_id in (
        "claude-haiku-5-5",
        "claude-opus-5-5",
        "claude-fable-5",
        "claude-sonnet-5-5",
    ):
        prices = pricing.for_model(model_id)
        assert prices is not None
        assert prices.cache_write == pytest.approx(prices.input * 1.25)


def test_for_model_dated_snapshot_and_unknown() -> None:
    pricing = AnthropicPricing()
    assert pricing.for_model("claude-haiku-4-5-20251001") == pricing.for_model(
        "claude-haiku-4-5"
    )
    assert pricing.for_model("gpt-unknown") is None
    assert pricing.for_model(None) is None


def test_from_mapping_overrides_defaults() -> None:
    pricing = AnthropicPricing.from_mapping(
        {
            "claude-opus-5-5": {
                "input": 1,
                "output": 2,
                "cache_read": 3,
                "cache_write": 4,
            }
        }
    )
    assert pricing.for_model("claude-opus-5-5") == ModelPrices(
        input=1, output=2, cache_read=3, cache_write=4
    )
    assert pricing.for_model("claude-haiku-5-5") is not None


def test_record_cost_with_provider_tag(monkeypatch: pytest.MonkeyPatch) -> None:
    import robotsix_llmio.anthropic.pricing as pricing_mod

    captured: dict = {}

    def _fake_record_cost(response, get_cost, *, provider=None):
        captured["provider"] = provider
        captured["cost"] = get_cost(response)

    monkeypatch.setattr(pricing_mod, "record_cost", _fake_record_cost)
    record_anthropic_cost(
        _usage(input_tokens=1_000_000), "claude-haiku-5-5", AnthropicPricing()
    )
    assert captured == {"provider": "anthropic", "cost": pytest.approx(0.10)}


def test_record_cost_unpriced_model_records_nothing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import robotsix_llmio.anthropic.pricing as pricing_mod

    calls: list = []
    monkeypatch.setattr(pricing_mod, "record_cost", lambda *a, **k: calls.append(a))
    record_anthropic_cost(_usage(input_tokens=10), "mystery", AnthropicPricing())
    assert calls == []


def test_lazy_exports_resolve() -> None:
    # AnthropicModel / AnthropicProvider import pydantic-ai's Anthropic
    # model, which needs the optional `anthropic` SDK ([anthropic] extra).
    pytest.importorskip("anthropic")
    import robotsix_llmio.anthropic as pkg

    for name in pkg.__all__:
        assert getattr(pkg, name) is not None
    with pytest.raises(AttributeError):
        pkg.does_not_exist  # noqa: B018

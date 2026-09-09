"""Offline unit tests for DeepSeek client-side cost computation + recording."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from robotsix_llmio.deepseek.pricing import (
    DeepseekPricing,
    ModelPrices,
    compute_cost,
    record_deepseek_cost,
)

_PRICES = ModelPrices(input=1.0, output=2.0, cache_read=0.1)


# --- compute_cost arithmetic ------------------------------------------------


def test_compute_cost_cache_hit_miss_split():
    """Cache-miss tokens bill at ``input``, cache-hit tokens at ``cache_read``,
    completion tokens at ``output``."""
    usage = SimpleNamespace(
        prompt_tokens=1000,
        completion_tokens=500,
        prompt_cache_hit_tokens=800,
        prompt_cache_miss_tokens=200,
        model_extra=None,
    )
    expected = (200 * 1.0 + 800 * 0.1 + 500 * 2.0) / 1_000_000
    assert compute_cost(usage, _PRICES) == pytest.approx(expected)


def test_compute_cost_no_cache_fields_treats_all_input_as_uncached():
    usage = SimpleNamespace(prompt_tokens=1000, completion_tokens=500, model_extra=None)
    expected = (1000 * 1.0 + 500 * 2.0) / 1_000_000
    assert compute_cost(usage, _PRICES) == pytest.approx(expected)


def test_compute_cost_falls_back_to_cached_tokens_detail():
    """When DeepSeek's hit/miss fields are absent, the OpenAI-style
    ``prompt_tokens_details.cached_tokens`` splits the input."""
    usage = SimpleNamespace(
        prompt_tokens=1000,
        completion_tokens=0,
        prompt_tokens_details=SimpleNamespace(cached_tokens=300),
        model_extra=None,
    )
    expected = (700 * 1.0 + 300 * 0.1) / 1_000_000
    assert compute_cost(usage, _PRICES) == pytest.approx(expected)


def test_compute_cost_reads_cache_fields_from_model_extra():
    """The hit/miss counters arrive as OpenAI-SDK extras (``model_extra``)."""
    usage = SimpleNamespace(
        prompt_tokens=1000,
        completion_tokens=0,
        prompt_cache_hit_tokens=None,
        prompt_cache_miss_tokens=None,
        model_extra={
            "prompt_cache_hit_tokens": 600,
            "prompt_cache_miss_tokens": 400,
        },
    )
    expected = (400 * 1.0 + 600 * 0.1) / 1_000_000
    assert compute_cost(usage, _PRICES) == pytest.approx(expected)


def test_compute_cost_none_usage_or_prices_returns_none():
    assert compute_cost(None, _PRICES) is None
    assert compute_cost(SimpleNamespace(prompt_tokens=1), None) is None


# --- pricing config ---------------------------------------------------------


def test_default_pricing_covers_chat_and_reasoner():
    pricing = DeepseekPricing()
    assert pricing.for_model("deepseek-chat") is not None
    assert pricing.for_model("deepseek-reasoner") is not None


def test_from_mapping_overrides_defaults_and_keeps_others():
    pricing = DeepseekPricing.from_mapping(
        {"deepseek-chat": {"input": 9.0, "output": 8.0, "cache_read": 0.5}}
    )
    chat = pricing.for_model("deepseek-chat")
    assert chat == ModelPrices(input=9.0, output=8.0, cache_read=0.5)
    # Unspecified model keeps its baked default.
    assert pricing.for_model("deepseek-reasoner") is not None


def test_for_model_unknown_returns_none():
    assert DeepseekPricing().for_model("no-such-model") is None
    assert DeepseekPricing().for_model(None) is None


# --- record_deepseek_cost ---------------------------------------------------


def test_record_deepseek_cost_calls_record_cost_with_provider(monkeypatch):
    """A priced model records via ``record_cost`` with ``provider='deepseek'``
    and the client-computed cost."""
    import robotsix_llmio.deepseek.pricing as pricing_mod

    captured: dict = {}

    def _fake_record_cost(response, get_cost, *, provider=None):
        captured["provider"] = provider
        captured["cost"] = get_cost(response)

    monkeypatch.setattr(pricing_mod, "record_cost", _fake_record_cost)

    usage = SimpleNamespace(
        prompt_tokens=1000,
        completion_tokens=500,
        prompt_cache_hit_tokens=800,
        prompt_cache_miss_tokens=200,
        model_extra=None,
    )
    response = SimpleNamespace(model="deepseek-chat", usage=usage)
    pricing = DeepseekPricing()
    record_deepseek_cost(response, pricing)

    prices = pricing.for_model("deepseek-chat")
    expected = (200 * prices.input + 800 * prices.cache_read + 500 * prices.output) / (
        1_000_000
    )
    assert captured["provider"] == "deepseek"
    assert captured["cost"] == pytest.approx(expected)


def test_record_deepseek_cost_unpriced_model_records_nothing(monkeypatch):
    """An unknown/unpriced model id records nothing and does not raise."""
    import robotsix_llmio.deepseek.pricing as pricing_mod

    calls: list = []
    monkeypatch.setattr(
        pricing_mod, "record_cost", lambda *a, **k: calls.append((a, k))
    )

    response = SimpleNamespace(
        model="some-unpriced-model",
        usage=SimpleNamespace(prompt_tokens=10, completion_tokens=5, model_extra=None),
    )
    record_deepseek_cost(response, DeepseekPricing())  # must not raise
    assert calls == []


def test_record_deepseek_cost_none_pricing_records_nothing(monkeypatch):
    import robotsix_llmio.deepseek.pricing as pricing_mod

    calls: list = []
    monkeypatch.setattr(
        pricing_mod, "record_cost", lambda *a, **k: calls.append((a, k))
    )
    response = SimpleNamespace(model="deepseek-chat", usage=SimpleNamespace())
    record_deepseek_cost(response, None)
    assert calls == []

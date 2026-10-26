"""Offline unit tests for the direct-API Anthropic provider + model.

Fully offline and key-free: key resolution, prefix resolution through the
factory, per-level settings, and an end-to-end ``request`` against a mocked
Messages API endpoint that checks the client-side cost is recorded.
"""

from __future__ import annotations

import asyncio
from typing import Any

import pytest

pytest.importorskip(
    "pydantic_ai.models.anthropic",
    reason="Anthropic transport requires the 'anthropic' extra",
    exc_type=ImportError,
)

import httpx2

from robotsix_llmio.anthropic import AnthropicAPIError
from robotsix_llmio.anthropic.model import AnthropicModel
from robotsix_llmio.anthropic.provider import AnthropicProvider
from robotsix_llmio.core.factory import get_provider_for_identifier
from robotsix_llmio.core.identifier import parse_model_identifier

# --- auth -------------------------------------------------------------------


def test_missing_key_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    with pytest.raises(AnthropicAPIError, match="Anthropic API key missing"):
        AnthropicProvider()


def test_env_key_and_default_base_url(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ANTHROPIC_API_KEY", "env-key")
    provider = AnthropicProvider()
    assert provider._api_key == "env-key"
    assert provider._base_url == "https://api.anthropic.com"


# --- identifier / factory ---------------------------------------------------


def test_identifier_parses_on_first_hyphen() -> None:
    parsed = parse_model_identifier("anthropic-claude-opus-5-5")
    assert parsed.provider == "anthropic"
    assert parsed.model_name == "claude-opus-5-5"


def test_identifier_resolves_to_anthropic_provider(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("LLMIO_PROVIDER", raising=False)
    provider = get_provider_for_identifier(
        "anthropic-claude-opus-5-5", api_key="test-key"
    )
    assert isinstance(provider, AnthropicProvider)


# --- model construction -----------------------------------------------------


def test_new_model_builds_cost_model() -> None:
    provider = AnthropicProvider(
        api_key="k", max_tokens=1234, effort_by_level={2: "high"}
    )
    model, http_client = provider.new_model(model="claude-opus-5-5", level=2)
    try:
        assert isinstance(model, AnthropicModel)
        assert model.model_name == "claude-opus-5-5"
        assert model.pricing is provider._pricing
        assert model.settings == {"max_tokens": 1234, "anthropic_effort": "high"}
    finally:
        asyncio.run(http_client.aclose())


def test_new_model_unmapped_level_keeps_api_default() -> None:
    provider = AnthropicProvider(api_key="k")
    model, http_client = provider.new_model(model="claude-haiku-5-5", level=1)
    try:
        assert model.settings is None
    finally:
        asyncio.run(http_client.aclose())


def test_new_model_requires_model() -> None:
    with pytest.raises(ValueError, match="model"):
        AnthropicProvider(api_key="k").new_model()


# --- end-to-end request against a mocked Messages API -----------------------


def test_request_records_client_side_cost(monkeypatch: pytest.MonkeyPatch) -> None:
    from pydantic_ai.messages import ModelRequest, UserPromptPart
    from pydantic_ai.models import ModelRequestParameters

    import robotsix_llmio.anthropic.pricing as pricing_mod
    import robotsix_llmio.anthropic.provider as provider_mod

    seen: dict[str, Any] = {}

    def handler(request: httpx2.Request) -> httpx2.Response:
        seen["url"] = str(request.url)
        seen["key"] = request.headers.get("x-api-key")
        return httpx2.Response(
            200,
            json={
                "id": "msg_1",
                "type": "message",
                "role": "assistant",
                "model": "claude-opus-5-5",
                "content": [{"type": "text", "text": "hi"}],
                "stop_reason": "end_turn",
                "stop_sequence": None,
                "usage": {
                    "input_tokens": 100,
                    "output_tokens": 50,
                    "cache_read_input_tokens": 1000,
                    "cache_creation_input_tokens": 200,
                },
            },
        )

    monkeypatch.setattr(
        provider_mod,
        "timeout_http_client",
        lambda: httpx2.AsyncClient(transport=httpx2.MockTransport(handler)),
    )
    captured: dict[str, Any] = {}

    def _fake_record_cost(response, get_cost, *, provider=None):
        captured["provider"] = provider
        captured["cost"] = get_cost(response)

    monkeypatch.setattr(pricing_mod, "record_cost", _fake_record_cost)

    # A small max_tokens keeps pydantic-ai on the non-streamed request path
    # (it streams behind the scenes for the model's default max output).
    provider = AnthropicProvider(api_key="secret", max_tokens=1024)
    model, http_client = provider.new_model(model="claude-opus-5-5", level=2)

    async def _run() -> Any:
        try:
            return await model.request(
                [ModelRequest(parts=[UserPromptPart(content="hello")])],
                None,
                ModelRequestParameters(),
            )
        finally:
            await http_client.aclose()

    response = asyncio.run(_run())

    assert seen["url"].startswith("https://api.anthropic.com/")
    assert seen["key"] == "secret"
    assert response.parts[0].content == "hi"
    prices = provider._pricing.for_model("claude-opus-5-5")
    assert prices is not None
    expected = (
        100 * prices.input
        + 1000 * prices.cache_read
        + 200 * prices.cache_write
        + 50 * prices.output
    ) / 1_000_000
    assert captured["provider"] == "anthropic"
    assert captured["cost"] == pytest.approx(expected)


# --- baked default chain: first slot dispatches to this transport -----------


@pytest.mark.parametrize("level", [1, 2, 3])
def test_baked_first_slot_dispatches_to_anthropic_transport(
    level: int, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The baked chain's position-0 slot reaches the real direct-API transport.

    Resolves through ``get_provider_for_level`` (the active slot after the
    autouse tracker reset is position 0) — no factory mocking — and checks
    the resolved provider is :class:`AnthropicProvider`, builds an
    :class:`AnthropicModel` for the level's model, and has baked pricing.
    """
    from robotsix_llmio.config.tier import TierConfig
    from robotsix_llmio.core.factory import get_provider_for_level

    monkeypatch.setenv("ANTHROPIC_API_KEY", "k")
    cfg = TierConfig()
    tlc = cfg.for_level(level)
    assert tlc is cfg.chain[0].for_level(level)
    assert tlc.provider == "anthropic"

    provider = get_provider_for_level(level, tier_config=cfg)
    assert isinstance(provider, AnthropicProvider)

    model, http_client = provider.new_model(model=tlc.model_name, level=level)
    try:
        assert isinstance(model, AnthropicModel)
        assert model.model_name == tlc.model_name
        assert provider._pricing.for_model(tlc.model_name) is not None
    finally:
        asyncio.run(http_client.aclose())

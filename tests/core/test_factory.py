"""Tests for the provider-agnostic factory (``core.factory``)."""

from __future__ import annotations

import pytest

from robotsix_llmio.core.factory import get_provider_for_identifier
from robotsix_llmio.core.identifier import MalformedIdentifierError

# -- new get_provider_for_identifier tests ----------------------------------


def test_identifier_claude_sdk_opus(monkeypatch: pytest.MonkeyPatch) -> None:
    """``claudeSDK-opus`` resolves to ClaudeSDKProvider."""
    monkeypatch.delenv("LLMIO_PROVIDER", raising=False)
    from robotsix_llmio.claude_sdk import ClaudeSDKProvider

    provider = get_provider_for_identifier("claudeSDK-opus")

    assert isinstance(provider, ClaudeSDKProvider)


def test_identifier_openrouter_deepseek(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``openrouter-deepseek/deepseek-v4-flash`` resolves to
    OpenRouterDeepseekProvider."""
    monkeypatch.delenv("LLMIO_PROVIDER", raising=False)
    from robotsix_llmio.openrouter._deepseek_provider import (
        OpenRouterDeepseekProvider,
    )

    provider = get_provider_for_identifier(
        "openrouter-deepseek/deepseek-v4-flash",
        api_key="test-key",
    )

    assert isinstance(provider, OpenRouterDeepseekProvider)


def test_identifier_forward_kwargs(monkeypatch: pytest.MonkeyPatch) -> None:
    """**kwargs are forwarded to the provider constructor."""
    monkeypatch.delenv("LLMIO_PROVIDER", raising=False)
    from robotsix_llmio.openrouter._deepseek_provider import (
        OpenRouterDeepseekProvider,
    )

    provider = get_provider_for_identifier(
        "openrouter-deepseek/deepseek-v4-flash",
        api_key="test-key",
        base_url="https://x",
    )

    assert isinstance(provider, OpenRouterDeepseekProvider)


def test_identifier_unknown_prefix_raises(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An unknown provider prefix raises ValueError listing known prefixes."""
    monkeypatch.delenv("LLMIO_PROVIDER", raising=False)

    with pytest.raises(ValueError) as excinfo:
        get_provider_for_identifier("bogusPrefix-opus")

    message = str(excinfo.value)
    assert "bogusPrefix" in message
    assert "claudeSDK" in message
    assert "openrouter" in message


def test_identifier_malformed_raises() -> None:
    """A malformed identifier raises MalformedIdentifierError."""
    with pytest.raises(MalformedIdentifierError):
        get_provider_for_identifier("no_hyphen_at_all")

    with pytest.raises(MalformedIdentifierError):
        get_provider_for_identifier("-emptyprefix")

    with pytest.raises(MalformedIdentifierError):
        get_provider_for_identifier("prefix-")

"""Tests for the provider-agnostic factory (``core.factory``)."""

from __future__ import annotations

import importlib

import pytest

from robotsix_llmio.core import factory, get_provider, register_provider
from robotsix_llmio.core.factory import get_provider_for_identifier
from robotsix_llmio.core.identifier import MalformedIdentifierError

# -- legacy get_provider tests -----------------------------------------------


def test_default_resolution(monkeypatch: pytest.MonkeyPatch) -> None:
    """No env var and no ``provider=`` resolves the default backend."""
    monkeypatch.delenv("LLMIO_PROVIDER", raising=False)
    from robotsix_llmio.openrouter_deepseek import OpenRouterDeepseekProvider

    provider = get_provider(api_key="x")

    assert isinstance(provider, OpenRouterDeepseekProvider)


def test_env_override(monkeypatch: pytest.MonkeyPatch) -> None:
    """``LLMIO_PROVIDER`` selects the backend when no argument is given."""
    monkeypatch.setenv("LLMIO_PROVIDER", "claude-sdk")
    from robotsix_llmio.claude_sdk import ClaudeSDKProvider

    provider = get_provider()

    assert isinstance(provider, ClaudeSDKProvider)


def test_explicit_argument_beats_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """An explicit ``provider=`` argument wins over the env var."""
    monkeypatch.setenv("LLMIO_PROVIDER", "openrouter-deepseek")
    from robotsix_llmio.claude_sdk import ClaudeSDKProvider

    provider = get_provider(provider="claude-sdk")

    assert isinstance(provider, ClaudeSDKProvider)


def test_unknown_name_lists_known(monkeypatch: pytest.MonkeyPatch) -> None:
    """An unregistered name raises an error listing the known names."""
    monkeypatch.delenv("LLMIO_PROVIDER", raising=False)

    with pytest.raises(ValueError) as excinfo:
        get_provider(provider="nope")

    message = str(excinfo.value)
    assert "nope" in message
    assert "openrouter-deepseek" in message
    assert "claude-sdk" in message


def test_missing_extra_is_actionable(monkeypatch: pytest.MonkeyPatch) -> None:
    """A backend whose module import fails raises an actionable ImportError."""
    monkeypatch.delenv("LLMIO_PROVIDER", raising=False)
    register_provider(
        "fake-broken",
        module="robotsix_llmio._does_not_exist",
        class_name="Nope",
        extra="fake_extra",
    )

    try:
        with pytest.raises(ImportError) as excinfo:
            get_provider(provider="fake-broken")
    finally:
        factory._PROVIDER_REGISTRY.pop("fake-broken", None)

    message = str(excinfo.value)
    assert "fake_extra" in message
    assert "pip install 'robotsix-llmio[fake_extra]'" in message


def test_import_module_failure_is_actionable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An ImportError from ``import_module`` is reframed with the extra name."""
    monkeypatch.delenv("LLMIO_PROVIDER", raising=False)

    def _boom(_name: str) -> object:
        raise ImportError("boom")

    monkeypatch.setattr(importlib, "import_module", _boom)

    with pytest.raises(ImportError) as excinfo:
        get_provider(provider="openrouter-deepseek")

    message = str(excinfo.value)
    assert "openrouter_deepseek" in message
    assert "pip install 'robotsix-llmio[openrouter_deepseek]'" in message


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
    """``openrouter[deepseek]-deepseek/deepseek-v4-flash`` resolves to
    OpenRouterDeepseekProvider."""
    monkeypatch.delenv("LLMIO_PROVIDER", raising=False)
    from robotsix_llmio.openrouter_deepseek import OpenRouterDeepseekProvider

    provider = get_provider_for_identifier(
        "openrouter[deepseek]-deepseek/deepseek-v4-flash",
        api_key="test-key",
    )

    assert isinstance(provider, OpenRouterDeepseekProvider)


def test_identifier_forward_kwargs(monkeypatch: pytest.MonkeyPatch) -> None:
    """**kwargs are forwarded to the provider constructor."""
    monkeypatch.delenv("LLMIO_PROVIDER", raising=False)
    from robotsix_llmio.openrouter_deepseek import OpenRouterDeepseekProvider

    provider = get_provider_for_identifier(
        "openrouter[deepseek]-deepseek/deepseek-v4-flash",
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

"""Offline unit tests for ``OpenRouterProvider.__init__`` auth resolution.

``OpenRouterProvider`` is a concrete class (no abstract methods), so
tests can instantiate a minimal subclass directly. ``__init__`` never
calls any hooks, so a plain subclass is sufficient. All tests are
fully offline and key-free — each test sets or clears ``OPENROUTER_API_KEY``
explicitly via ``monkeypatch``.
"""

from __future__ import annotations

import sys
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from robotsix_llmio.core.provider import Tier
from robotsix_llmio.openrouter.provider import OpenRouterProvider


class _Concrete(OpenRouterProvider):
    """Minimal concrete provider so the class can be instantiated in tests.

    ``OpenRouterProvider`` is now a concrete class — no abstract methods
    to implement.
    """


def test_missing_key_raises(monkeypatch):
    """With no explicit key and no env var, construction raises a clear
    ``RuntimeError`` naming the missing OpenRouter API key."""
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    with pytest.raises(RuntimeError, match="OpenRouter API key missing"):
        _Concrete(api_key=None)


def test_explicit_api_key_succeeds(monkeypatch):
    """An explicit ``api_key=`` is stored even when the env var is unset, and
    the default ``base_url`` is recorded."""
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    provider = _Concrete(api_key="sk-test")
    assert provider._api_key == "sk-test"
    assert provider._base_url == "https://openrouter.ai/api/v1"


def test_env_var_fallback_succeeds(monkeypatch):
    """When ``api_key`` is ``None`` the constructor falls back to the
    ``OPENROUTER_API_KEY`` environment variable."""
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-env")
    provider = _Concrete(api_key=None)
    assert provider._api_key == "sk-env"


# ---------------------------------------------------------------------------
# Tests for ``new_model()``
# ---------------------------------------------------------------------------


class _NewModelProvider(OpenRouterProvider):
    """Concrete provider for ``new_model`` tests.

    Overrides every hook so the test can control inputs and observe side
    effects without touching the network or pydantic-ai internals.
    """

    _tier_compat: dict = {Tier.DEFAULT: "test-model-default", Tier.CHEAP: "cheap-model"}  # noqa: RUF012

    def __init__(self, *, api_key: str = "sk-test"):
        super().__init__(api_key=api_key)
        self._post_build_calls: list[tuple] = []
        self._model_cls_mock = MagicMock()

    def _model_class(self):
        return self._model_cls_mock

    def _post_build_model(self, model, level: int):
        self._post_build_calls.append((model, level))


def _install_fake_pydantic_openrouter(monkeypatch) -> MagicMock:
    """Inject a fake ``pydantic_ai.providers.openrouter`` module so the
    lazy import inside ``new_model()`` succeeds.

    Returns the mock ``OpenRouterProvider`` class so tests can assert it
    was called with the expected arguments.
    """
    mock_provider_cls = MagicMock()
    fake_openrouter = SimpleNamespace(OpenRouterProvider=mock_provider_cls)
    for key, val in [
        ("pydantic_ai", SimpleNamespace()),
        ("pydantic_ai.providers", SimpleNamespace()),
        ("pydantic_ai.providers.openrouter", fake_openrouter),
    ]:
        sys.modules[key] = val
        monkeypatch.setitem(sys.modules, key, val)
    return mock_provider_cls


def test_new_model_returns_model_and_http_client(monkeypatch):
    """``new_model()`` returns a ``(model, http_client)`` 2-tuple."""
    _install_fake_pydantic_openrouter(monkeypatch)
    mock_http = MagicMock()
    monkeypatch.setattr(
        "robotsix_llmio.openrouter.provider.timeout_http_client",
        lambda: mock_http,
    )

    provider = _NewModelProvider()
    model, http_client = provider.new_model(model="test-model-default")

    assert http_client is mock_http
    assert model is not None


def test_new_model_passes_correct_model_name(monkeypatch):
    """``new_model()`` passes the given model name as the first positional
    argument to the model-class constructor."""
    _install_fake_pydantic_openrouter(monkeypatch)
    mock_http = MagicMock()
    monkeypatch.setattr(
        "robotsix_llmio.openrouter.provider.timeout_http_client",
        lambda: mock_http,
    )

    provider = _NewModelProvider()
    provider.new_model(model="test-model-default")

    provider._model_cls_mock.assert_called_once()
    args = provider._model_cls_mock.call_args[0]
    assert args[0] == "test-model-default"


def test_new_model_returns_http_client_from_timeout_client(monkeypatch):
    """The ``http_client`` returned by ``new_model()`` is the exact object
    created by ``timeout_http_client()``."""
    _install_fake_pydantic_openrouter(monkeypatch)
    mock_http = MagicMock()
    monkeypatch.setattr(
        "robotsix_llmio.openrouter.provider.timeout_http_client",
        lambda: mock_http,
    )

    provider = _NewModelProvider()
    _, http_client = provider.new_model(model="test-model-default")

    assert http_client is mock_http


def test_new_model_default_base_url_uses_api_key_path(monkeypatch):
    """With the default ``base_url``, ``new_model()`` constructs the pydantic-ai
    provider from ``api_key``/``http_client`` and does not build an
    ``openai_client``."""
    mock_provider_cls = _install_fake_pydantic_openrouter(monkeypatch)
    mock_http = MagicMock()
    monkeypatch.setattr(
        "robotsix_llmio.openrouter.provider.timeout_http_client",
        lambda: mock_http,
    )

    provider = _NewModelProvider()
    provider.new_model(model="test-model-default")

    _, kwargs = mock_provider_cls.call_args
    assert kwargs["api_key"] == "sk-test"
    assert kwargs["http_client"] is mock_http
    assert "openai_client" not in kwargs


def test_new_model_custom_base_url_builds_openai_client(monkeypatch):
    """A custom ``base_url`` is wired into an ``AsyncOpenAI`` client that is
    passed to the pydantic-ai provider via ``openai_client=``."""
    mock_provider_cls = _install_fake_pydantic_openrouter(monkeypatch)
    mock_http = MagicMock()
    monkeypatch.setattr(
        "robotsix_llmio.openrouter.provider.timeout_http_client",
        lambda: mock_http,
    )

    mock_openai_cls = MagicMock()
    fake_openai = SimpleNamespace(AsyncOpenAI=mock_openai_cls)
    monkeypatch.setitem(sys.modules, "openai", fake_openai)

    provider = _NewModelProvider()
    provider._base_url = "https://proxy.example/api/v1"
    provider.new_model(model="test-model-default")

    _, openai_kwargs = mock_openai_cls.call_args
    assert openai_kwargs["base_url"] == "https://proxy.example/api/v1"
    assert openai_kwargs["api_key"] == "sk-test"
    assert openai_kwargs["http_client"] is mock_http

    _, prov_kwargs = mock_provider_cls.call_args
    assert prov_kwargs["openai_client"] is mock_openai_cls.return_value


def test_new_model_calls_post_build_model(monkeypatch):
    """``new_model()`` invokes ``_post_build_model`` with the constructed
    model and the level (0 when called via the deprecated tier path)."""
    _install_fake_pydantic_openrouter(monkeypatch)
    mock_http = MagicMock()
    monkeypatch.setattr(
        "robotsix_llmio.openrouter.provider.timeout_http_client",
        lambda: mock_http,
    )

    provider = _NewModelProvider()
    with pytest.warns(DeprecationWarning):
        model, _ = provider.new_model(tier=Tier.CHEAP)

    assert len(provider._post_build_calls) == 1
    called_model, called_level = provider._post_build_calls[0]
    assert called_model is model
    assert called_level == 0

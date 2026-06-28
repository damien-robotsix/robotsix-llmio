"""Shared fixtures for config-package tests."""

from __future__ import annotations

import os
from unittest.mock import MagicMock

import pytest

from robotsix_llmio.core.provider import LLMProvider


@pytest.fixture(autouse=True)
def clean_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """Remove all LLMIO_* variables so tests start from a known-clean env."""
    for key in tuple(os.environ):
        if key.startswith("LLMIO_"):
            monkeypatch.delenv(key, raising=False)


@pytest.fixture
def mock_get_provider_for_identifier(
    monkeypatch: pytest.MonkeyPatch,
) -> MagicMock:
    """Patch ``get_provider_for_identifier`` as referenced by the factory module."""
    mock = MagicMock(return_value=MagicMock(spec=LLMProvider))
    monkeypatch.setattr(
        "robotsix_llmio.config.factory.get_provider_for_identifier", mock
    )
    return mock

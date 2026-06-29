"""Tests for the provider-model identifier parser (``core.identifier``)."""

from __future__ import annotations

import pytest

from robotsix_llmio.core.identifier import (
    MalformedIdentifierError,
    ParsedIdentifier,
    parse_model_identifier,
)
from robotsix_llmio.exceptions import RobotsixLLMIOError


class TestParseModelIdentifier:
    """Happy-path and edge-case parsing."""

    def test_claude_sdk_opus(self):
        result = parse_model_identifier("claudeSDK-opus")
        assert result == ParsedIdentifier(provider="claudeSDK", model_name="opus")

    def test_openrouter_deepseek_with_model(self):
        result = parse_model_identifier("openrouter-deepseek/deepseek-v4-flash")
        assert result == ParsedIdentifier(
            provider="openrouter",
            model_name="deepseek/deepseek-v4-flash",
        )

    def test_model_name_contains_hyphens(self):
        result = parse_model_identifier("claudeSDK-some-model-v2")
        assert result == ParsedIdentifier(
            provider="claudeSDK", model_name="some-model-v2"
        )

    def test_model_name_contains_slashes(self):
        result = parse_model_identifier("openrouter-deepseek/deepseek-v4-pro")
        assert result.model_name == "deepseek/deepseek-v4-pro"

    def test_provider_split_on_first_hyphen(self):
        result = parse_model_identifier("openrouter-deepseek/deepseek-v4-flash")
        assert result.provider == "openrouter"
        assert result.model_name == "deepseek/deepseek-v4-flash"


class TestMalformedIdentifiers:
    """Malformed inputs raise MalformedIdentifierError."""

    def test_no_hyphen(self):
        with pytest.raises(MalformedIdentifierError, match="No hyphen-delimited"):
            parse_model_identifier("no_hyphen_at_all")

    def test_empty_provider(self):
        with pytest.raises(MalformedIdentifierError, match="Empty provider prefix"):
            parse_model_identifier("-onlymodel")

    def test_empty_model(self):
        with pytest.raises(MalformedIdentifierError, match="Empty model name"):
            parse_model_identifier("claudeSDK-")


class TestMalformedIdentifierError:
    """The exception class hierarchy."""

    def test_inherits_from_robotsix_llmio_error(self):
        assert issubclass(MalformedIdentifierError, RobotsixLLMIOError)

    def test_message_contains_identifier(self):
        with pytest.raises(MalformedIdentifierError) as exc:
            parse_model_identifier("no_hyphen_at_all")
        assert "no_hyphen_at_all" in str(exc.value)


class TestParsedIdentifier:
    """The result type."""

    def test_is_tuple_subclass(self):
        p = parse_model_identifier("claudeSDK-opus")
        assert isinstance(p, tuple)
        assert len(p) == 2

    def test_field_access(self):
        p = parse_model_identifier("openrouter-deepseek/deepseek-v4-flash")
        assert p.provider == "openrouter"
        assert p.model_name == "deepseek/deepseek-v4-flash"

    def test_unpacking(self):
        provider, model_name = parse_model_identifier("claudeSDK-opus")
        assert provider == "claudeSDK"
        assert model_name == "opus"

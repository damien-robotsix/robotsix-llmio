"""Tests for Claude SDK usage mapping and per-model camelCase aggregation.

Extracted from tests/claude_sdk/test_claude_sdk.py.
"""

from __future__ import annotations

from typing import ClassVar

import pytest

pytest.importorskip("pydantic_ai")

from robotsix_llmio.claude_sdk._usage import (
    _aggregate_per_model,
    _best_usage_dict,
    map_usage_dict,
)
from robotsix_llmio.claude_sdk.model import _map_usage

# --- usage mapping ---------------------------------------------------------


def test_map_usage_from_result():
    class _R:
        usage: ClassVar = {
            "input_tokens": 10,
            "output_tokens": 5,
            "cache_read_input_tokens": 3,
            "cache_creation_input_tokens": 7,
        }

    u = _map_usage(_R())
    assert (u.input_tokens, u.output_tokens) == (10, 5)
    assert (u.cache_read_tokens, u.cache_write_tokens) == (3, 7)


def test_map_usage_handles_none_and_partial():
    assert _map_usage(None).input_tokens == 0

    class _R:
        usage: ClassVar = {"input_tokens": 4}

    assert _map_usage(_R()).output_tokens == 0


def test_map_usage_dict_full():
    u = map_usage_dict(
        {
            "input_tokens": 10,
            "output_tokens": 5,
            "cache_read_input_tokens": 3,
            "cache_creation_input_tokens": 7,
        }
    )
    assert (u.input_tokens, u.output_tokens) == (10, 5)
    # key renames: cache_read_input_tokens -> cache_read_tokens,
    # cache_creation_input_tokens -> cache_write_tokens
    assert (u.cache_read_tokens, u.cache_write_tokens) == (3, 7)


def test_map_usage_dict_partial_defaults_to_zero():
    u = map_usage_dict({"input_tokens": 4})
    assert u.input_tokens == 4
    assert (u.output_tokens, u.cache_read_tokens, u.cache_write_tokens) == (0, 0, 0)


def test_map_usage_dict_empty():
    u = map_usage_dict({})
    assert (u.input_tokens, u.output_tokens) == (0, 0)
    assert (u.cache_read_tokens, u.cache_write_tokens) == (0, 0)


def test_map_usage_dict_none():
    u = map_usage_dict(None)
    assert (u.input_tokens, u.output_tokens) == (0, 0)
    assert (u.cache_read_tokens, u.cache_write_tokens) == (0, 0)


def test_map_usage_dict_non_dict():
    for bad in (["input_tokens", 1], "input_tokens=1"):
        u = map_usage_dict(bad)
        assert (u.input_tokens, u.output_tokens) == (0, 0)
        assert (u.cache_read_tokens, u.cache_write_tokens) == (0, 0)


# --- per-model camelCase aggregation ---------------------------------------


def test_aggregate_per_model_single_model():

    d = {
        "claude-3-5-haiku-20241022": {
            "inputTokens": 200,
            "outputTokens": 50,
            "cacheReadInputTokens": 10,
            "cacheCreationInputTokens": 0,
        }
    }
    out = _aggregate_per_model(d)
    assert out == {
        "input_tokens": 200,
        "output_tokens": 50,
        "cache_read_input_tokens": 10,
        "cache_creation_input_tokens": 0,
    }


def test_aggregate_per_model_multi_model():

    d = {
        "claude-3-5-haiku-20241022": {"inputTokens": 200, "outputTokens": 50},
        "claude-3-5-sonnet-20241022": {"inputTokens": 100, "outputTokens": 30},
    }
    out = _aggregate_per_model(d)
    assert out == {"input_tokens": 300, "output_tokens": 80}


def test_aggregate_per_model_partial_keys():

    d = {
        "claude-3-5-haiku-20241022": {"inputTokens": 200},
    }
    out = _aggregate_per_model(d)
    assert out == {"input_tokens": 200}


def test_aggregate_per_model_not_per_model_format():

    # Values are not dicts → not the per-model format.
    assert _aggregate_per_model({"input_tokens": 4}) is None
    assert _aggregate_per_model({"a": 1, "b": 2}) is None


def test_aggregate_per_model_empty():

    assert _aggregate_per_model({}) is None


def test_best_usage_dict_prefers_model_usage_per_model():
    """``_best_usage_dict`` aggregates the per-model camelCase
    ``model_usage`` and returns flat snake_case, ignoring ``usage``."""

    class _R:
        model_usage: ClassVar = {
            "claude-3-5-haiku-20241022": {"inputTokens": 200, "outputTokens": 50},
        }
        usage: ClassVar = {"input_tokens": 999, "output_tokens": 999}  # ignored

    out = _best_usage_dict(_R)
    assert out == {"input_tokens": 200, "output_tokens": 50}


def test_best_usage_dict_falls_back_to_usage():
    """When ``model_usage`` is absent, ``_best_usage_dict`` uses the flat
    ``usage`` dict."""

    class _R:
        usage: ClassVar = {"input_tokens": 10, "output_tokens": 5}

    out = _best_usage_dict(_R)
    assert out == {"input_tokens": 10, "output_tokens": 5}


def test_best_usage_dict_model_usage_empty_falls_through():
    """When ``model_usage`` is an empty dict, ``_best_usage_dict`` falls
    through to ``usage``."""

    class _R:
        model_usage: ClassVar = {}
        usage: ClassVar = {"input_tokens": 42, "output_tokens": 17}

    out = _best_usage_dict(_R)
    assert out == {"input_tokens": 42, "output_tokens": 17}


def test_map_usage_dict_from_per_model():
    """End-to-end: ``map_usage_dict`` receives the aggregated flat dict and
    returns correct ``RequestUsage``."""
    from robotsix_llmio.claude_sdk._usage import map_usage_dict

    d = {
        "claude-3-5-haiku-20241022": {
            "inputTokens": 200,
            "outputTokens": 50,
            "cacheReadInputTokens": 10,
            "cacheCreationInputTokens": 7,
        }
    }
    u = map_usage_dict(_aggregate_per_model(d))
    assert (u.input_tokens, u.output_tokens) == (200, 50)
    assert (u.cache_read_tokens, u.cache_write_tokens) == (10, 7)

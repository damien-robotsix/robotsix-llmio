"""OpenRouter transport layer — cost extraction, usage.include, transient."""

from __future__ import annotations

import json
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from robotsix_llmio.openrouter.model import (
    PROVIDER_NAME,
    OpenRouterModel,
    _CostCapturingStream,
    _get_cost_from_response,
    _inject_usage_include,
    _resolve_model_settings,
    record_openrouter_cost,
)
from robotsix_llmio.openrouter.transient import (
    is_openrouter_transient,
    is_openrouter_upstream_error,
)


def test_get_cost_from_usage_attr():
    resp = SimpleNamespace(usage=SimpleNamespace(cost=0.0123, model_extra=None))
    assert _get_cost_from_response(resp) == 0.0123


def test_get_cost_from_model_extra():
    resp = SimpleNamespace(usage=SimpleNamespace(model_extra={"cost": 0.5}))
    assert _get_cost_from_response(resp) == 0.5


def test_get_cost_none_when_absent():
    assert _get_cost_from_response(SimpleNamespace(usage=None)) is None
    assert (
        _get_cost_from_response(
            SimpleNamespace(usage=SimpleNamespace(cost=None, model_extra=None))
        )
        is None
    )


def test_inject_usage_include_sets_flag():
    ms: dict = {}
    _inject_usage_include((), {"model_settings": ms})
    assert ms["extra_body"]["usage"]["include"] is True


def test_inject_usage_include_preserves_existing_extra_body():
    ms = {"extra_body": {"provider": {"only": ["X"]}}}
    _inject_usage_include((), {"model_settings": ms})
    assert ms["extra_body"]["provider"]["only"] == ["X"]
    assert ms["extra_body"]["usage"]["include"] is True


def test_resolve_model_settings_from_args_position():
    ms = {"k": 1}
    assert _resolve_model_settings(("messages", False, ms, "params"), {}) is ms


def test_upstream_error_detected():
    class ValidationError(Exception):
        pass

    e = ValidationError("finish_reason expected one of ... got 'error'")
    assert is_openrouter_upstream_error(e) is True
    assert is_openrouter_transient(e) is True


def test_plain_validation_error_not_upstream():
    class ValidationError(Exception):
        pass

    assert is_openrouter_upstream_error(ValidationError("bad schema")) is False


@patch("robotsix_llmio.openrouter.model.get_recording_span")
def test_record_cost_noop_when_cost_none(mock_get_span):
    resp = SimpleNamespace(usage=None)
    assert record_openrouter_cost(resp) is None
    mock_get_span.assert_not_called()


@patch("robotsix_llmio.openrouter.model.get_recording_span")
def test_record_cost_noop_when_span_none(mock_get_span):
    mock_get_span.return_value = None
    resp = SimpleNamespace(usage=SimpleNamespace(cost=0.01, model_extra=None))
    assert record_openrouter_cost(resp) is None
    mock_get_span.assert_called_once()


@patch("robotsix_llmio.openrouter.model.get_recording_span")
def test_record_cost_always_set_attributes(mock_get_span):
    span = MagicMock()
    mock_get_span.return_value = span
    resp = SimpleNamespace(usage=SimpleNamespace(cost=0.02, model_extra=None))
    record_openrouter_cost(resp)
    span.set_attribute.assert_any_call("gen_ai.usage.cost", 0.02)
    span.set_attribute.assert_any_call(
        "langfuse.observation.cost_details", json.dumps({"total": 0.02})
    )
    span.set_attribute.assert_any_call("gen_ai.operation.name", "chat")
    span.set_attribute.assert_any_call("gen_ai.provider.name", PROVIDER_NAME)
    span.set_attribute.assert_any_call("gen_ai.system", PROVIDER_NAME)
    span.set_attribute.assert_any_call(
        "langfuse.observation.metadata.provider", PROVIDER_NAME
    )


@patch("robotsix_llmio.openrouter.model.get_recording_span")
def test_record_cost_model_and_token_attributes(mock_get_span):
    span = MagicMock()
    mock_get_span.return_value = span
    resp = SimpleNamespace(
        usage=SimpleNamespace(
            cost=0.03,
            model_extra=None,
            prompt_tokens=10,
            completion_tokens=20,
        ),
        model="x/y",
    )
    record_openrouter_cost(resp)
    span.set_attribute.assert_any_call("gen_ai.request.model", "x/y")
    span.set_attribute.assert_any_call("gen_ai.usage.input_tokens", 10)
    span.set_attribute.assert_any_call("gen_ai.usage.output_tokens", 20)


@patch("robotsix_llmio.openrouter.model.get_recording_span")
def test_record_cost_cached_tokens_dict_details(mock_get_span):
    span = MagicMock()
    mock_get_span.return_value = span
    resp = SimpleNamespace(
        usage=SimpleNamespace(
            cost=0.04,
            model_extra=None,
            prompt_tokens_details={
                "cached_tokens": 5,
                "cache_creation_input_tokens": 3,
            },
        ),
    )
    record_openrouter_cost(resp)
    span.set_attribute.assert_any_call("gen_ai.usage.cache_read_input_tokens", 5)
    span.set_attribute.assert_any_call("gen_ai.usage.cache_creation_input_tokens", 3)


@patch("robotsix_llmio.openrouter.model.get_recording_span")
def test_record_cost_reasoning_tokens_attr_details(mock_get_span):
    span = MagicMock()
    mock_get_span.return_value = span
    resp = SimpleNamespace(
        usage=SimpleNamespace(
            cost=0.05,
            model_extra=None,
            completion_tokens_details=SimpleNamespace(reasoning_tokens=7),
        ),
    )
    record_openrouter_cost(resp)
    span.set_attribute.assert_any_call("gen_ai.usage.reasoning_tokens", 7)


@patch("robotsix_llmio.openrouter.model.get_recording_span")
def test_record_cost_absent_optional_fields_skipped(mock_get_span):
    span = MagicMock()
    mock_get_span.return_value = span
    resp = SimpleNamespace(usage=SimpleNamespace(cost=0.06, model_extra=None))
    record_openrouter_cost(resp)
    recorded = {c.args[0] for c in span.set_attribute.call_args_list}
    assert "gen_ai.usage.cost" in recorded
    assert "langfuse.observation.cost_details" in recorded
    assert "gen_ai.operation.name" in recorded
    assert "gen_ai.provider.name" in recorded
    assert "gen_ai.system" in recorded
    assert "langfuse.observation.metadata.provider" in recorded
    assert "gen_ai.request.model" not in recorded
    assert "gen_ai.usage.input_tokens" not in recorded
    assert "gen_ai.usage.output_tokens" not in recorded
    assert "gen_ai.usage.cache_read_input_tokens" not in recorded
    assert "gen_ai.usage.reasoning_tokens" not in recorded


# ---------------------------------------------------------------------------
# _FakeStream helper for _CostCapturingStream tests
# ---------------------------------------------------------------------------


class _FakeStream:
    """Minimal AsyncStream stand-in for tests — no network, no openai."""

    def __init__(self, chunks: list) -> None:
        self._chunks = iter(chunks)

    async def __aenter__(self) -> _FakeStream:
        return self

    async def __aexit__(self, *a: object) -> None:
        pass

    def __aiter__(self) -> _FakeStream:
        return self

    async def __anext__(self):
        try:
            return next(self._chunks)
        except StopIteration:
            raise StopAsyncIteration from None


# ---------------------------------------------------------------------------
# _CostCapturingStream tests
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
@patch("robotsix_llmio.openrouter.model.get_recording_span")
async def test_cost_capturing_stream_records_cost_on_exhaustion(mock_get_span):
    span = MagicMock()
    mock_get_span.return_value = span

    chunks = [
        SimpleNamespace(usage=None),
        SimpleNamespace(usage=None),
        SimpleNamespace(
            usage=SimpleNamespace(
                cost=0.007,
                model_extra=None,
                prompt_tokens=10,
                completion_tokens=5,
            ),
            model="x/y",
        ),
    ]
    stream = _CostCapturingStream(_FakeStream(chunks))

    collected: list = []
    async for chunk in stream:
        collected.append(chunk)

    assert len(collected) == 3
    mock_get_span.assert_called_once()
    span.set_attribute.assert_any_call("gen_ai.usage.cost", 0.007)


@pytest.mark.asyncio
@patch("robotsix_llmio.openrouter.model.get_recording_span")
async def test_cost_capturing_stream_noop_when_no_usage_chunk(mock_get_span):
    chunks = [
        SimpleNamespace(usage=None),
        SimpleNamespace(usage=None),
    ]
    stream = _CostCapturingStream(_FakeStream(chunks))

    collected: list = []
    async for chunk in stream:
        collected.append(chunk)

    assert len(collected) == 2
    mock_get_span.assert_not_called()


@pytest.mark.asyncio
async def test_cost_capturing_stream_aenter_aexit_delegate():
    inner = MagicMock()
    inner.__aenter__ = AsyncMock(return_value=inner)
    inner.__aexit__ = AsyncMock(return_value=None)

    stream = _CostCapturingStream(inner)
    result = await stream.__aenter__()
    assert result is stream
    inner.__aenter__.assert_called_once()

    await stream.__aexit__(None, None, None)
    inner.__aexit__.assert_called_once()


@pytest.mark.asyncio
@patch("robotsix_llmio.openrouter.model.isinstance", return_value=True)
@patch("robotsix_llmio.openrouter.model.OpenAIChatModel.__init__", return_value=None)
@patch("robotsix_llmio.openrouter.model.OpenAIChatModel._completions_create")
async def test_completions_create_stream_returns_capturing_wrapper(
    mock_super, mock_init, mock_isinstance
):
    mock_super.return_value = _FakeStream([])

    model = OpenRouterModel("x/y")
    result = await model._completions_create([], True, {}, {})

    assert isinstance(result, _CostCapturingStream)


@pytest.mark.asyncio
@patch("robotsix_llmio.openrouter.model.OpenAIChatModel.__init__", return_value=None)
@patch("robotsix_llmio.openrouter.model.OpenAIChatModel._completions_create")
@patch("robotsix_llmio.openrouter.model.get_recording_span")
async def test_completions_create_non_stream_records_cost_directly(
    mock_get_span, mock_super, mock_init
):
    span = MagicMock()
    mock_get_span.return_value = span
    mock_super.return_value = SimpleNamespace(
        usage=SimpleNamespace(cost=0.01, model_extra=None)
    )

    model = OpenRouterModel("x/y")
    result = await model._completions_create([], False, {}, {})

    assert not isinstance(result, _CostCapturingStream)
    span.set_attribute.assert_any_call("gen_ai.usage.cost", 0.01)

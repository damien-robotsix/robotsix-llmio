"""OpenRouter transport layer — cost extraction, usage.include, transient."""

from __future__ import annotations

import asyncio
import json
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

from robotsix_llmio.openrouter.model import (
    _CACHE_CONTROL_MARKER,
    PROVIDER_NAME,
    OpenRouterModel,
    _CostCapturingStream,
    _get_cost_from_response,
    _inject_usage_include,
    _resolve_model_settings,
    record_openrouter_cost,
)
from robotsix_llmio.openrouter.transient import (
    is_deepseek_reasoning_400,
    is_openrouter_transient,
    is_openrouter_upstream_error,
    is_openrouter_upstream_payment_error,
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


# --- upstream provider 402 (vs. our own credits running out) ---------------


class _HTTPErr(Exception):
    """Stand-in for ``ModelHTTPError``: carries ``status_code`` and renders the
    body in its ``str()``, which is what the predicate inspects."""

    def __init__(self, status_code: int, body: str) -> None:
        super().__init__(f"status_code: {status_code}, body: {body}")
        self.status_code = status_code


#: The real body seen on 2026-07-29 (Python dict repr, via ModelHTTPError).
_UPSTREAM_402_BODY = (
    "{'message': 'Provider returned error', 'code': 402, 'metadata': "
    '{\'raw\': \'{"error":{"message":"Insufficient Balance"}}\', '
    "'provider_name': 'DeepSeek', 'is_byok': False}}"
)


def test_upstream_provider_402_is_transient():
    """OpenRouter's *upstream* provider had no balance — retrying lets it route
    to one of the other providers serving the same model."""
    e = _HTTPErr(402, _UPSTREAM_402_BODY)
    assert is_openrouter_upstream_payment_error(e) is True
    assert is_openrouter_transient(e) is True


def test_upstream_provider_402_detected_in_json_rendering():
    """Same signal, JSON-rendered rather than Python-repr'd."""
    body = (
        '{"message":"Provider returned error","code":402,"metadata":'
        '{"provider_name":"DeepSeek","is_byok":false}}'
    )
    assert is_openrouter_upstream_payment_error(_HTTPErr(402, body)) is True


def test_own_credits_402_is_not_transient():
    """Our OpenRouter account being out of credits must fail fast — retrying
    cannot help, and silently riding it out would burn the retry budget."""
    e = _HTTPErr(402, "{'error': {'message': 'Insufficient credits', 'code': 402}}")
    assert is_openrouter_upstream_payment_error(e) is False
    assert is_openrouter_transient(e) is False


def test_byok_402_is_not_transient():
    """A BYOK key is *ours*: its provider running dry is a real billing failure
    on our side, so it must not be retried."""
    body = (
        "{'message': 'Provider returned error', 'code': 402, 'metadata': "
        "{'provider_name': 'DeepSeek', 'is_byok': True}}"
    )
    assert is_openrouter_upstream_payment_error(_HTTPErr(402, body)) is False


def test_non_402_with_provider_metadata_is_not_payment_error():
    """The status gate matters: a 500 carrying provider metadata is already
    transient via the core set, not via the payment predicate."""
    assert is_openrouter_upstream_payment_error(_HTTPErr(500, _UPSTREAM_402_BODY)) is (
        False
    )


# --- DeepSeek reasoning-400 --------------------------------------------------


_DEEPSEEK_400_BODY = (
    "Error code: 400 - {'error': {'message': "
    '"The `reasoning_content` in the thinking mode must be passed back '
    "to the API.\", 'type': 'invalid_request_error'}}"
)


def test_deepseek_reasoning_400_is_detected():
    """HTTP 400 with the distinctive reasoning_content + thinking mode markers
    is transient — a re-run with reasoning injected will succeed."""
    e = _HTTPErr(400, _DEEPSEEK_400_BODY)
    assert is_deepseek_reasoning_400(e) is True
    assert is_openrouter_transient(e) is True


def test_deepseek_reasoning_400_in_cause_chain():
    """The reasoning-400 may be wrapped by a framework exception — it should
    still be recognised through the cause chain."""

    class Wrapper(Exception):
        pass

    inner = _HTTPErr(400, _DEEPSEEK_400_BODY)
    outer = Wrapper("wrapped")
    outer.__cause__ = inner
    assert is_deepseek_reasoning_400(outer) is False  # the wrapper itself isn't a 400
    assert is_openrouter_transient(outer) is True  # but the chain finds it


def test_plain_400_is_not_reasoning_400():
    """A generic 400 (e.g. bad schema) must NOT be mistaken for the
    DeepSeek reasoning-content error."""
    e = _HTTPErr(400, "Bad request: invalid schema")
    assert is_deepseek_reasoning_400(e) is False
    assert is_openrouter_transient(e) is False


def test_non_400_is_not_reasoning_400():
    """Status other than 400 must not match, even with the keywords."""
    e = _HTTPErr(500, _DEEPSEEK_400_BODY)
    assert is_deepseek_reasoning_400(e) is False


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


@patch("robotsix_llmio.openrouter.model.get_recording_span")
def test_cost_capturing_stream_records_cost_on_exhaustion(mock_get_span):
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

    async def _run():
        collected: list = []
        async for chunk in stream:
            collected.append(chunk)
        return collected

    collected = asyncio.run(_run())
    assert len(collected) == 3
    mock_get_span.assert_called_once()
    span.set_attribute.assert_any_call("gen_ai.usage.cost", 0.007)


@patch("robotsix_llmio.openrouter.model.get_recording_span")
def test_cost_capturing_stream_noop_when_no_usage_chunk(mock_get_span):
    chunks = [
        SimpleNamespace(usage=None),
        SimpleNamespace(usage=None),
    ]
    stream = _CostCapturingStream(_FakeStream(chunks))

    async def _run():
        collected: list = []
        async for chunk in stream:
            collected.append(chunk)
        return collected

    collected = asyncio.run(_run())
    assert len(collected) == 2
    mock_get_span.assert_not_called()


def test_cost_capturing_stream_aenter_aexit_delegate():
    inner = MagicMock()
    inner.__aenter__ = AsyncMock(return_value=inner)
    inner.__aexit__ = AsyncMock(return_value=None)

    async def _run():
        stream = _CostCapturingStream(inner)
        result = await stream.__aenter__()
        assert result is stream
        inner.__aenter__.assert_called_once()

        await stream.__aexit__(None, None, None)
        inner.__aexit__.assert_called_once()

    asyncio.run(_run())


@patch("robotsix_llmio.openrouter.model.isinstance", return_value=True)
@patch("robotsix_llmio.openrouter.model.OpenAIChatModel.__init__", return_value=None)
@patch("robotsix_llmio.openrouter.model.OpenAIChatModel._completions_create")
def test_completions_create_stream_returns_capturing_wrapper(
    mock_super, mock_init, mock_isinstance
):
    mock_super.return_value = _FakeStream([])

    async def _run():
        model = OpenRouterModel("x/y")
        return await model._completions_create([], True, {}, {})

    result = asyncio.run(_run())
    assert isinstance(result, _CostCapturingStream)


@patch("robotsix_llmio.openrouter.model.OpenAIChatModel.__init__", return_value=None)
@patch("robotsix_llmio.openrouter.model.OpenAIChatModel._completions_create")
@patch("robotsix_llmio.openrouter.model.get_recording_span")
def test_completions_create_non_stream_records_cost_directly(
    mock_get_span, mock_super, mock_init
):
    span = MagicMock()
    mock_get_span.return_value = span
    mock_super.return_value = SimpleNamespace(
        usage=SimpleNamespace(cost=0.01, model_extra=None)
    )

    async def _run():
        model = OpenRouterModel("x/y")
        return await model._completions_create([], False, {}, {})

    result = asyncio.run(_run())
    assert not isinstance(result, _CostCapturingStream)
    span.set_attribute.assert_any_call("gen_ai.usage.cost", 0.01)


# ---------------------------------------------------------------------------
# Prompt-caching tests — _map_messages, _get_tool_choice, cache logging
# ---------------------------------------------------------------------------


@patch("robotsix_llmio.openrouter.model.OpenAIChatModel.__init__", return_value=None)
@patch("robotsix_llmio.openrouter.model.OpenAIChatModel._map_messages")
def test_map_messages_adds_cache_control_to_last_system_message(mock_super, mock_init):
    mock_super.return_value = [
        {"role": "system", "content": "first system"},
        {"role": "user", "content": "hello"},
        {"role": "system", "content": "second system"},
        {"role": "assistant", "content": "hi"},
    ]

    async def _run():
        model = OpenRouterModel("x/y")
        return await model._map_messages([], {})

    result = asyncio.run(_run())

    # Only the last system message should carry cache_control.
    assert result[0].get("cache_control") is None
    assert result[1].get("cache_control") is None
    assert result[2]["cache_control"] == _CACHE_CONTROL_MARKER
    assert result[3].get("cache_control") is None


@patch("robotsix_llmio.openrouter.model.OpenAIChatModel.__init__", return_value=None)
@patch("robotsix_llmio.openrouter.model.OpenAIChatModel._map_messages")
def test_map_messages_no_cache_control_when_disabled(mock_super, mock_init):
    mock_super.return_value = [
        {"role": "system", "content": "only system"},
    ]

    async def _run():
        model = OpenRouterModel("x/y")
        model._prompt_caching_enabled = False
        return await model._map_messages([], {})

    result = asyncio.run(_run())
    assert result[0].get("cache_control") is None


@patch("robotsix_llmio.openrouter.model.OpenAIChatModel.__init__", return_value=None)
@patch("robotsix_llmio.openrouter.model.OpenAIChatModel._map_messages")
def test_map_messages_no_system_messages_no_cache_control(mock_super, mock_init):
    mock_super.return_value = [
        {"role": "user", "content": "hello"},
    ]

    async def _run():
        model = OpenRouterModel("x/y")
        return await model._map_messages([], {})

    result = asyncio.run(_run())
    assert result[0].get("cache_control") is None


@patch("robotsix_llmio.openrouter.model.OpenAIChatModel.__init__", return_value=None)
@patch("robotsix_llmio.openrouter.model.OpenAIChatModel._get_tool_choice")
def test_get_tool_choice_adds_cache_control_to_last_tool(mock_super, mock_init):
    mock_super.return_value = (
        [
            {"type": "function", "function": {"name": "tool_a", "description": "a"}},
            {"type": "function", "function": {"name": "tool_b", "description": "b"}},
        ],
        "auto",
    )

    model = OpenRouterModel("x/y")
    tools, _tool_choice = model._get_tool_choice({}, {})

    assert len(tools) == 2
    assert tools[0].get("cache_control") is None
    assert tools[1]["cache_control"] == _CACHE_CONTROL_MARKER


@patch("robotsix_llmio.openrouter.model.OpenAIChatModel.__init__", return_value=None)
@patch("robotsix_llmio.openrouter.model.OpenAIChatModel._get_tool_choice")
def test_get_tool_choice_no_cache_control_when_disabled(mock_super, mock_init):
    mock_super.return_value = (
        [{"type": "function", "function": {"name": "tool_a", "description": "a"}}],
        "auto",
    )

    model = OpenRouterModel("x/y")
    model._prompt_caching_enabled = False
    tools, _tool_choice = model._get_tool_choice({}, {})

    assert tools[0].get("cache_control") is None


@patch("robotsix_llmio.openrouter.model.OpenAIChatModel.__init__", return_value=None)
@patch("robotsix_llmio.openrouter.model.OpenAIChatModel._get_tool_choice")
def test_get_tool_choice_no_tools_no_cache_control(mock_super, mock_init):
    mock_super.return_value = ([], None)

    model = OpenRouterModel("x/y")
    tools, _tool_choice = model._get_tool_choice({}, {})

    assert tools == []


@patch("robotsix_llmio.openrouter.model.logger")
@patch("robotsix_llmio.openrouter.model.get_recording_span")
def test_record_cost_logs_cache_hit_ratio(mock_get_span, mock_logger):
    span = MagicMock()
    mock_get_span.return_value = span
    resp = SimpleNamespace(
        usage=SimpleNamespace(
            cost=0.04,
            model_extra=None,
            prompt_tokens=100,
            prompt_tokens_details={
                "cached_tokens": 80,
                "cache_creation_input_tokens": 20,
            },
        ),
        model="deepseek/deepseek-v4-pro",
    )
    record_openrouter_cost(resp)
    mock_logger.info.assert_called_once()
    args, _kwargs = mock_logger.info.call_args
    log_msg = args[0] % args[1:]
    assert "prompt tokens: total=100 cached=80" in log_msg
    assert "80.0% hit" in log_msg
    assert "cache_creation=20" in log_msg


@patch("robotsix_llmio.openrouter.model.logger")
@patch("robotsix_llmio.openrouter.model.get_recording_span")
def test_record_cost_no_cache_log_when_no_cached_tokens(mock_get_span, mock_logger):
    span = MagicMock()
    mock_get_span.return_value = span
    resp = SimpleNamespace(
        usage=SimpleNamespace(
            cost=0.04,
            model_extra=None,
            prompt_tokens=100,
        ),
        model="x/y",
    )
    record_openrouter_cost(resp)
    # No log call for cache because cached_tokens is absent.
    assert not mock_logger.info.called

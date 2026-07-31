"""Tracing tests for no-tools ``request()`` span attributes — regression
coverage ensuring identity, token, and usage attributes are stamped
correctly even when no tools are attached."""

from __future__ import annotations

import asyncio

import pytest

pytest.importorskip("pydantic_ai")

from pydantic_ai.messages import (
    ModelRequest,
    UserPromptPart,
)
from pydantic_ai.models import ModelRequestParameters

from robotsix_llmio.claude_sdk._model import ClaudeSDKModel
from robotsix_llmio.core.tracing import (
    GEN_AI_OPERATION_NAME,
    GEN_AI_PROVIDER_NAME,
    GEN_AI_REQUEST_MODEL,
    GEN_AI_SYSTEM,
    GEN_AI_USAGE_INPUT_TOKENS,
    GEN_AI_USAGE_OUTPUT_TOKENS,
    OP_CHAT,
)

from .conftest import (
    _install_fake_sdk,
)

# --- no-tools request() span attributes (regression) -----------------------


def test_notools_request_stamps_gen_ai_attributes(monkeypatch, otel_exporter_tracer):
    """After ``ClaudeSDKModel.request()`` runs with a recording span active, that
    span carries provider.model identity, system, and token-usage attributes
    — independently of whether cost was recorded, and consistent with the
    tool-loop path."""
    exporter, tracer = otel_exporter_tracer

    fake = _install_fake_sdk(monkeypatch)

    async def _fake_query(*, prompt, options):
        yield fake.AssistantMessage("hello world")
        yield fake.ResultMessage({"input_tokens": 10, "output_tokens": 5})

    fake.query = _fake_query

    model = ClaudeSDKModel("opus")

    async def _run():
        with tracer.start_as_current_span("root"):
            return await model.request(
                [ModelRequest(parts=[UserPromptPart(content="hi")])],
                model_settings=None,
                model_request_parameters=ModelRequestParameters(output_mode="text"),
            )

    asyncio.run(_run())

    spans = exporter.get_finished_spans()
    assert len(spans) == 1, f"expected 1 span, got {[s.name for s in spans]}"
    attrs = spans[0].attributes
    assert attrs[GEN_AI_OPERATION_NAME] == OP_CHAT
    assert attrs[GEN_AI_PROVIDER_NAME] == "claude-sdk"
    assert attrs[GEN_AI_SYSTEM] == "anthropic"
    assert attrs[GEN_AI_REQUEST_MODEL] == "opus"
    assert attrs[GEN_AI_USAGE_INPUT_TOKENS] == 10
    assert attrs[GEN_AI_USAGE_OUTPUT_TOKENS] == 5


def test_notools_request_stamps_tokens_from_model_usage(
    monkeypatch,
    otel_exporter_tracer,
):
    """When the SDK result carries token data in ``model_usage`` rather than
    ``usage``, the generation span still records non-zero tokens."""
    exporter, tracer = otel_exporter_tracer

    fake = _install_fake_sdk(monkeypatch)

    async def _fake_query(*, prompt, options):
        yield fake.AssistantMessage("hello world")
        yield fake.ResultMessage(model_usage={"input_tokens": 30, "output_tokens": 12})

    fake.query = _fake_query

    model = ClaudeSDKModel("opus")

    async def _run():
        with tracer.start_as_current_span("root"):
            return await model.request(
                [ModelRequest(parts=[UserPromptPart(content="hi")])],
                model_settings=None,
                model_request_parameters=ModelRequestParameters(output_mode="text"),
            )

    resp = asyncio.run(_run())
    # The pydantic-ai usage should also reflect model_usage tokens.
    assert resp.usage.input_tokens == 30
    assert resp.usage.output_tokens == 12

    spans = exporter.get_finished_spans()
    assert len(spans) == 1, f"expected 1 span, got {[s.name for s in spans]}"
    attrs = spans[0].attributes
    assert attrs[GEN_AI_USAGE_INPUT_TOKENS] == 30
    assert attrs[GEN_AI_USAGE_OUTPUT_TOKENS] == 12


def test_notools_request_skips_missing_usage(monkeypatch, otel_exporter_tracer):
    """When the SDK ResultMessage has no usage dict, token attributes are not set
    (no spurious zeros), but identity attributes are still stamped."""
    exporter, tracer = otel_exporter_tracer

    fake = _install_fake_sdk(monkeypatch)

    async def _fake_query(*, prompt, options):
        yield fake.AssistantMessage("hello")
        yield fake.ResultMessage()  # no usage

    fake.query = _fake_query

    model = ClaudeSDKModel("sonnet")

    async def _run():
        with tracer.start_as_current_span("root"):
            return await model.request(
                [ModelRequest(parts=[UserPromptPart(content="hi")])],
                model_settings=None,
                model_request_parameters=ModelRequestParameters(output_mode="text"),
            )

    asyncio.run(_run())

    spans = exporter.get_finished_spans()
    assert len(spans) == 1
    attrs = spans[0].attributes
    # Identity always set.
    assert attrs[GEN_AI_PROVIDER_NAME] == "claude-sdk"
    assert attrs[GEN_AI_SYSTEM] == "anthropic"
    assert attrs[GEN_AI_REQUEST_MODEL] == "sonnet"
    # Token keys absent entirely.
    assert GEN_AI_USAGE_INPUT_TOKENS not in attrs
    assert GEN_AI_USAGE_OUTPUT_TOKENS not in attrs


def test_no_span_recording_is_noop(monkeypatch):
    """When no span is recording (OpenTelemetry absent or no TracerProvider),
    ``request()`` behaves exactly as before — no exception, no writes."""
    fake = _install_fake_sdk(monkeypatch)

    async def _fake_query(*, prompt, options):
        yield fake.AssistantMessage("ok")
        yield fake.ResultMessage({"input_tokens": 1, "output_tokens": 1})

    fake.query = _fake_query

    model = ClaudeSDKModel("haiku")
    # No OTel tracer installed — get_recording_span() returns None.
    response = asyncio.run(
        model.request(
            [ModelRequest(parts=[UserPromptPart(content="hi")])],
            model_settings=None,
            model_request_parameters=ModelRequestParameters(output_mode="text"),
        )
    )
    assert response.model_name == "haiku"
    assert response.parts[0].content == "ok"

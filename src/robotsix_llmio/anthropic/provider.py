"""Direct-API Anthropic transport provider — auth + per-level model construction.

A sibling of :class:`~robotsix_llmio.claude_sdk.ClaudeSDKProvider` that serves
Claude models through the first-party Anthropic Messages API
(``https://api.anthropic.com``) with an ``ANTHROPIC_API_KEY`` instead of the
``claude`` CLI subscription seat. The model is pydantic-ai's native
``AnthropicModel`` (our :class:`~robotsix_llmio.anthropic.model.AnthropicModel`
subclass, which records cost client-side).

Thinking policy mirrors ``claude_sdk``: no per-level override is applied by
default, so each model runs with the API's own thinking default (adaptive
thinking on current Claude models). An optional ``effort_by_level`` mapping
sets ``output_config.effort`` per capability level where an operator wants to
tune reasoning depth.
"""

from __future__ import annotations

import os
from collections.abc import Mapping
from typing import Any, cast

from ..core import timeout_http_client
from ..core.http import _close_async_client
from ..core.provider import LLMProvider
from . import AnthropicAPIError
from .model import AnthropicModel
from .pricing import AnthropicPricing

#: Environment variable name for the Anthropic API key.
_ENV_ANTHROPIC_API_KEY = "ANTHROPIC_API_KEY"

#: The first-party Anthropic Messages API endpoint.
_DEFAULT_BASE_URL = "https://api.anthropic.com"


def _coerce_pricing(
    pricing: AnthropicPricing | Mapping[str, Any] | None,
) -> AnthropicPricing:
    """Normalise a ``pricing`` constructor arg into an :class:`AnthropicPricing`."""
    if pricing is None:
        return AnthropicPricing()
    if isinstance(pricing, AnthropicPricing):
        return pricing
    return AnthropicPricing.from_mapping(pricing)


class AnthropicProvider(LLMProvider):
    """Builds cost-instrumented first-party Anthropic API models."""

    def __init__(
        self,
        *,
        api_key: str | None = None,
        base_url: str = _DEFAULT_BASE_URL,
        max_tokens: int | None = None,
        pricing: AnthropicPricing | Mapping[str, Any] | None = None,
        effort_by_level: Mapping[int, str] | None = None,
    ) -> None:
        """Configure auth and the Anthropic REST endpoint.

        Args:
            api_key: Anthropic API key. Falls back to ``ANTHROPIC_API_KEY`` when
                omitted. The raw key is never written to any config file.
            base_url: Messages API endpoint. Defaults to the first-party API;
                pass a custom value to route through a proxy.
            max_tokens: Optional output token cap, forwarded to the model as
                ``max_tokens`` in the default model settings.
            pricing: Per-1M-token sticker prices used to compute cost
                client-side. An :class:`AnthropicPricing` or a model-id →
                ``{"input", "output", "cache_read", "cache_write"}`` mapping,
                merged over the baked defaults.
            effort_by_level: Optional capability-level → effort
                (``"low"``/``"medium"``/``"high"``/``"xhigh"``/``"max"``)
                mapping, sent as ``anthropic_effort``. Unmapped levels keep the
                API default (as ``claude_sdk`` does).

        """
        self._api_key = api_key or os.environ.get(_ENV_ANTHROPIC_API_KEY, "")
        if not self._api_key:
            raise AnthropicAPIError(
                "Anthropic API key missing: pass api_key= or set"
                f" {_ENV_ANTHROPIC_API_KEY}."
            )
        self._base_url = base_url
        self._max_tokens = max_tokens
        self._pricing = _coerce_pricing(pricing)
        self._effort_by_level = dict(effort_by_level or {})

    def _model_settings(self, level: int) -> dict[str, Any]:
        """Default model settings for *level* (``max_tokens`` / effort)."""
        settings: dict[str, Any] = {}
        if self._max_tokens is not None:
            settings["max_tokens"] = self._max_tokens
        effort = self._effort_by_level.get(level)
        if effort is not None:
            settings["anthropic_effort"] = effort
        return settings

    def new_model(
        self,
        *,
        model: str | None = None,
        level: int = 0,
    ) -> tuple[Any, Any]:
        """Build a model, returning ``(model, http_client)``.

        Args:
            model: The concrete Anthropic model id (e.g. ``"claude-opus-5-5"``).
            level: Capability level (1, 2, 3) used to look up
                ``effort_by_level``. ``0`` means unknown / direct call.

        The returned ``http_client`` is the timeout-configured client backing
        the model; the caller owns closing it.

        """
        from pydantic_ai.models.anthropic import AnthropicModelSettings
        from pydantic_ai.providers.anthropic import (
            AnthropicProvider as _PydAnthropicProvider,
        )

        if model is None:
            raise ValueError("`model` must be provided to `new_model()`.")

        http_client = timeout_http_client()
        try:
            pyd_provider = _PydAnthropicProvider(
                api_key=self._api_key,
                base_url=self._base_url,
                http_client=http_client,
            )
            settings = self._model_settings(level)
            model_obj = AnthropicModel(
                model,
                provider=pyd_provider,
                settings=cast(AnthropicModelSettings, settings) if settings else None,
            )
            model_obj.pricing = self._pricing
            return model_obj, http_client
        except BaseException:
            _close_async_client(http_client)
            raise

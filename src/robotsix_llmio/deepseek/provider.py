"""Direct-API DeepSeek transport provider — auth + per-level model construction.

A sibling of :class:`~robotsix_llmio.openrouter.provider.OpenRouterProvider`
that targets the DeepSeek API directly (``https://api.deepseek.com``) with a
DeepSeek-issued key. DeepSeek's API is OpenAI-compatible, so the model is a
plain :class:`~pydantic_ai.models.openai.OpenAIChatModel` (our
:class:`~robotsix_llmio.deepseek.model.DeepseekModel`) built on an
``AsyncOpenAI`` client pointed at the DeepSeek base URL — no OpenRouter-only
machinery (provider-routing kwargs, ``usage.include``, server ``usage.cost``).

Thinking policy is per level: level 1 resolves to the non-thinking model
(``deepseek-chat``) and level ≥ 2 to the thinking model (``deepseek-reasoner``),
because DeepSeek controls thinking primarily by model id on the direct API.
"""

from __future__ import annotations

import os
from collections.abc import Mapping
from typing import Any

from ..core import timeout_http_client
from ..core.http import _close_async_client
from ..core.provider import LLMProvider
from . import DeepseekAPIError
from .model import NON_THINKING_MODEL, THINKING_MODEL, DeepseekModel
from .pricing import DeepseekPricing

#: Environment variable name for the DeepSeek API key.
_ENV_DEEPSEEK_API_KEY = "DEEPSEEK_API_KEY"

#: The direct DeepSeek API endpoint (OpenAI-compatible).
_DEFAULT_BASE_URL = "https://api.deepseek.com"


def _coerce_pricing(
    pricing: DeepseekPricing | Mapping[str, Any] | None,
) -> DeepseekPricing:
    """Normalise a ``pricing`` constructor arg into a :class:`DeepseekPricing`.

    Accepts a ready :class:`DeepseekPricing`, a model-id → prices mapping (as it
    would arrive from a tier's ``provider_kwargs``), or ``None`` for the baked
    defaults.
    """
    if pricing is None:
        return DeepseekPricing()
    if isinstance(pricing, DeepseekPricing):
        return pricing
    return DeepseekPricing.from_mapping(pricing)


class DeepseekProvider(LLMProvider):
    """Builds cost-instrumented direct-API DeepSeek models from a model name."""

    def __init__(
        self,
        *,
        api_key: str | None = None,
        base_url: str = _DEFAULT_BASE_URL,
        max_tokens: int | None = None,
        pricing: DeepseekPricing | Mapping[str, Any] | None = None,
    ) -> None:
        """Configure auth and the DeepSeek REST endpoint.

        Args:
            api_key: DeepSeek API key. Falls back to ``DEEPSEEK_API_KEY`` when
                omitted. The raw key is never written to any config file.
            base_url: DeepSeek-compatible REST endpoint. Defaults to the public
                DeepSeek API; pass a custom value to route through a proxy.
            max_tokens: Optional output token cap, forwarded to the model as
                ``max_tokens`` in the default model settings.
            pricing: Per-1M-token sticker prices used to compute cost
                client-side (DeepSeek returns no cost). A
                :class:`~robotsix_llmio.deepseek.pricing.DeepseekPricing` or a
                model-id → ``{"input", "output", "cache_read"}`` mapping;
                merged over the baked defaults. Routed from a tier's
                ``provider_kwargs`` so prices are updatable without code changes.

        """
        self._api_key = api_key or os.environ.get(_ENV_DEEPSEEK_API_KEY, "")
        if not self._api_key:
            raise DeepseekAPIError(
                "DeepSeek API key missing: pass api_key= or set"
                f" {_ENV_DEEPSEEK_API_KEY}."
            )
        self._base_url = base_url
        self._max_tokens = max_tokens
        self._pricing = _coerce_pricing(pricing)

    # --- hooks for derived layers -------------------------------------------

    def _model_class(self) -> type[DeepseekModel]:
        """The DeepseekModel subclass to instantiate (overridable)."""
        return DeepseekModel

    def _post_build_model(self, model: DeepseekModel, level: int) -> None:
        """Stamp per-level thinking policy and pricing onto a fresh model.

        DeepSeek selects thinking vs non-thinking by model id on the direct API,
        so the policy resolves the model id per level: level 1 → the
        non-thinking model, level ≥ 2 → the thinking model. ``level == 0`` is
        the sentinel for a direct ``new_model()`` call and keeps the configured
        model id unchanged (a safe default).

        TODO: DeepSeek currently toggles thinking purely by model id
        (``deepseek-chat`` vs ``deepseek-reasoner``); it exposes no request
        parameter for it on the OpenAI-compatible endpoint. If a future API adds
        one, set it here instead of swapping the model id.
        """
        model.pricing = self._pricing
        if level == 1:
            model._model_name = NON_THINKING_MODEL
        elif level >= 2:
            model._model_name = THINKING_MODEL

    # --- core API -----------------------------------------------------------
    def new_model(
        self,
        *,
        model: str | None = None,
        level: int = 0,
    ) -> tuple[Any, Any]:
        """Build a model, returning ``(model, http_client)``.

        Args:
            model: The concrete DeepSeek model id (e.g. ``"deepseek-chat"`` or
                ``"deepseek-reasoner"``).
            level: Capability level (1, 2, 3) forwarded to
                :meth:`_post_build_model` for the per-level thinking policy.
                ``0`` means unknown / direct ``new_model()`` call.

        The returned ``http_client`` is the timeout-configured client backing
        the model; the caller owns closing it.

        """
        from openai import AsyncOpenAI
        from pydantic_ai.providers.openai import OpenAIProvider

        if model is None:
            raise ValueError("`model` must be provided to `new_model()`.")

        http_client = timeout_http_client()
        try:
            openai_client = AsyncOpenAI(
                base_url=self._base_url,
                api_key=self._api_key,
                http_client=http_client,
            )
            pyd_provider = OpenAIProvider(openai_client=openai_client)

            model_kwargs: dict[str, Any] = {}
            if self._max_tokens is not None:
                model_kwargs["settings"] = {"max_tokens": self._max_tokens}
            model_obj = self._model_class()(
                model, provider=pyd_provider, **model_kwargs
            )
            self._post_build_model(model_obj, level)
            return model_obj, http_client
        except BaseException:
            _close_async_client(http_client)
            raise

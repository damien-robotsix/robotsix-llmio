"""OpenRouter transport provider — auth + per-tier model construction.

Model-family agnostic. A derived layer may override :meth:`_model_class`
or :meth:`_post_build_model` to add provider-family quirks (pin, reasoning
policy, …).  Model names are resolved through :class:`TierConfig` (passed
via ``build_agent()``) or supplied directly to :meth:`new_model`.
"""

from __future__ import annotations

import os
from typing import Any

from robotsix_llmio.openrouter import OpenRouterAPIError

from ..core import timeout_http_client
from ..core.http import _close_async_client
from ..core.provider import LLMProvider
from ._base import _DEFAULT_BASE_URL
from .model import OpenRouterModel
from .transient import is_openrouter_transient

#: Environment variable name for the OpenRouter API key.
_ENV_OPENROUTER_API_KEY = "OPENROUTER_API_KEY"


class OpenRouterProvider(LLMProvider):
    """Builds cost-instrumented OpenRouter models from a model name.

    Subclasses MAY override :meth:`_model_class` / :meth:`_post_build_model`
    to add provider-family quirks (pin, reasoning policy, …).
    """

    def __init__(
        self,
        *,
        api_key: str | None = None,
        base_url: str = _DEFAULT_BASE_URL,
        max_tokens: int | None = None,
    ) -> None:
        """Configure auth and the OpenRouter REST endpoint.

        Args:
            api_key: OpenRouter API key. Falls back to ``OPENROUTER_API_KEY``
                when omitted.
            base_url: OpenRouter-compatible REST endpoint. Defaults to the
                public OpenRouter API; pass a custom value to route requests
                through a proxy or mirror.
            max_tokens: Optional output token cap (unused by the OpenRouter
                transport; accepted for tier-config compatibility).

        """
        self._api_key = api_key or os.environ.get(_ENV_OPENROUTER_API_KEY, "")
        if not self._api_key:
            raise OpenRouterAPIError(
                "OpenRouter API key missing: pass api_key= or set"
                f" {_ENV_OPENROUTER_API_KEY}."
            )
        self._base_url = base_url
        self._max_tokens = max_tokens

    # --- hooks for derived layers -------------------------------------------

    def _model_class(self) -> type[OpenRouterModel]:
        """The OpenRouterModel subclass to instantiate (overridable)."""
        return OpenRouterModel

    def _post_build_model(self, model: OpenRouterModel, level: int) -> None:
        """Hook to stamp per-level policy onto a freshly built model.

        *level* is the capability level (1, 2, or 3) rather than a ``Tier``
        enum, so subclasses can apply per-level policy directly.

        No-op in the base class.
        """

    # --- core API -----------------------------------------------------------
    def new_model(
        self,
        *,
        model: str | None = None,
        level: int = 0,
    ) -> tuple[Any, Any]:
        """Build a model, returning ``(model, http_client)``.

        Args:
            model: The concrete model name (e.g.
                ``"deepseek/deepseek-v4-flash"``).
            level: Capability level (1, 2, 3) forwarded to
                :meth:`_post_build_model` for per-level policy hooks.
                ``0`` means unknown / direct ``new_model()`` call.

        The returned ``http_client`` is the timeout-configured client backing
        the model; the caller owns closing it.

        """
        from pydantic_ai.providers.openrouter import (
            OpenRouterProvider as _PydOpenRouterProvider,
        )

        if model is None:
            raise ValueError("`model` must be provided to `new_model()`.")

        model_name = model
        http_client = timeout_http_client()
        try:
            if self._base_url == _DEFAULT_BASE_URL:
                pyd_provider = _PydOpenRouterProvider(
                    api_key=self._api_key, http_client=http_client
                )
            else:
                from openai import AsyncOpenAI

                openai_client = AsyncOpenAI(
                    base_url=self._base_url,
                    api_key=self._api_key,
                    http_client=http_client,
                )
                pyd_provider = _PydOpenRouterProvider(openai_client=openai_client)
            model_obj = self._model_class()(model_name, provider=pyd_provider)
            self._post_build_model(model_obj, level)
            return model_obj, http_client
        except BaseException:
            _close_async_client(http_client)
            raise

    def _is_transient(self, exc: BaseException) -> bool:
        return is_openrouter_transient(exc)

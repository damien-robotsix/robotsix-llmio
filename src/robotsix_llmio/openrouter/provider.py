"""OpenRouter transport provider — auth + per-tier model construction.

Model-family agnostic. A derived layer may override :meth:`_model_class`
or :meth:`_post_build_model` to add provider-family quirks (pin, reasoning
policy, …).  Model names are resolved through :class:`TierConfig` (passed
via ``build_agent()``) or supplied directly to :meth:`new_model`.
"""

from __future__ import annotations

import os
import warnings
from typing import Any, ClassVar

from ..core import timeout_http_client
from ..core.provider import LLMProvider, Tier
from ._base import _DEFAULT_BASE_URL
from .model import OpenRouterModel
from .transient import is_openrouter_transient


class OpenRouterProvider(LLMProvider):
    """Builds cost-instrumented OpenRouter models from a model name.

    Subclasses MAY override :meth:`_model_class` / :meth:`_post_build_model`
    to add provider-family quirks (pin, reasoning policy, …) and SHOULD
    set :attr:`_tier_compat` (or override :meth:`new_model`) to support the
    deprecated ``tier=`` fallback path.
    """

    # Minimal internal compat dict for the deprecated ``tier=`` path.
    # Subclasses populate this (or override ``new_model`` entirely).
    _tier_compat: ClassVar[dict[Tier, str]] = {}

    def __init__(
        self,
        *,
        api_key: str | None = None,
        base_url: str = _DEFAULT_BASE_URL,
    ) -> None:
        """Configure auth and the OpenRouter REST endpoint.

        Args:
            api_key: OpenRouter API key. Falls back to ``OPENROUTER_API_KEY``
                when omitted.
            base_url: OpenRouter-compatible REST endpoint. Defaults to the
                public OpenRouter API; pass a custom value to route requests
                through a proxy or mirror.
        """
        self._api_key = api_key or os.environ.get("OPENROUTER_API_KEY", "")
        if not self._api_key:
            raise RuntimeError(
                "OpenRouter API key missing: pass api_key= or set OPENROUTER_API_KEY."
            )
        self._base_url = base_url

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
        tier: Tier | None = None,
        level: int = 0,
    ) -> tuple[Any, Any]:
        """Build a model, returning ``(model, http_client)``.

        Parameters
        ----------
        model:
            **Primary** — the concrete model name.  When provided the model
            is constructed directly; *tier* is ignored.
        tier:
            **Deprecated** — use *model* instead.  When *model* is ``None``
            and *tier* is provided, resolves via a minimal internal compat
            dict and emits a :exc:`DeprecationWarning`.
        level:
            Capability level (1, 2, 3) forwarded to
            :meth:`_post_build_model` for per-level policy hooks.  ``0``
            means unknown / direct ``new_model()`` call.

        The returned ``http_client`` is the timeout-configured client backing
        the model; the caller owns closing it.
        """
        from pydantic_ai.providers.openrouter import (
            OpenRouterProvider as _PydOpenRouterProvider,
        )

        if model is not None:
            model_name = model
        elif tier is not None:
            warnings.warn(
                "The `tier` parameter on `new_model()` is deprecated. "
                "Pass `model=` with a concrete model name instead.",
                DeprecationWarning,
                stacklevel=2,
            )
            model_name = self._tier_compat[tier]
        else:
            raise ValueError(
                "Either `model` or `tier` must be provided to `new_model()`."
            )

        http_client = timeout_http_client()
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

    def _is_transient(self, exc: BaseException) -> bool:
        return is_openrouter_transient(exc)

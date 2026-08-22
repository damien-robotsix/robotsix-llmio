"""Derived DeepSeek-on-OpenRouter provider — the layer consumers plug in.

Model names are resolved from :class:`~robotsix_llmio.config.tier.TierConfig`
(via :meth:`~robotsix_llmio.core.provider.LLMProvider.build_agent`) or passed
directly to :meth:`new_model`.

Provider routing (which upstream serves a ``deepseek/*`` model, whether
OpenRouter may fall back, and what that fallback may cost) is configurable via
the constructor, so it can be set from tier config's ``provider_kwargs``
without a code change — see :class:`OpenRouterDeepseekProvider`.
"""

from __future__ import annotations

from ._deepseek_model import (
    DEFAULT_IGNORE_CAPABLE,
    DEFAULT_MAX_PRICE_CAPABLE,
    DEFAULT_MAX_PRICE_CHEAP,
    OpenRouterDeepseekModel,
    build_provider_routing,
)
from .provider import OpenRouterProvider


class OpenRouterDeepseekProvider(OpenRouterProvider):
    """OpenRouter preferring DeepSeek, with per-level reasoning + routing policy.

    Routing defaults keep DeepSeek first (prompt-cache warmth) but allow
    OpenRouter to fall back to another provider under a price ceiling, so a
    single upstream's outage cannot block every request. Override per tier from
    config, e.g.::

        {"level2": {"model": "openrouter-deepseek/deepseek-v4-pro",
                    "provider_kwargs": {"max_price_prompt": 1.0,
                                        "max_price_completion": 2.0,
                                        "ignore_providers": ["SomeProvider"]}}}
    """

    def __init__(
        self,
        *,
        preferred_provider: str | None = "DeepSeek",
        allow_fallbacks: bool = True,
        max_price_prompt: float | None = None,
        max_price_completion: float | None = None,
        ignore_providers: list[str] | None = None,
        **kwargs: object,
    ) -> None:
        """Configure auth (see :class:`OpenRouterProvider`) plus routing policy.

        Args:
            preferred_provider: Upstream tried first. ``None`` lets OpenRouter
                choose freely.
            allow_fallbacks: Whether OpenRouter may route past the preferred
                provider when it fails. Setting this ``False`` restores the old
                hard-pin behaviour and makes that provider a single point of
                failure.
            max_price_prompt: Prompt price ceiling, USD per 1M tokens. Falls
                back to the per-level default when omitted.
            max_price_completion: Completion price ceiling, USD per 1M tokens.
                Falls back to the per-level default when omitted.
            ignore_providers: Upstream providers to exclude from fallback
                routing (``provider.ignore``) — e.g. ones whose cache-read
                rate is a multiple of the preferred provider's. Falls back to
                the per-level default when omitted.
            **kwargs: Forwarded verbatim to :class:`OpenRouterProvider`
                (``api_key``, ``base_url``, ``max_tokens``).

        """
        super().__init__(**kwargs)  # type: ignore[arg-type]
        self._preferred_provider = preferred_provider
        self._allow_fallbacks = allow_fallbacks
        self._max_price_prompt = max_price_prompt
        self._max_price_completion = max_price_completion
        self._ignore_providers = ignore_providers

    def _model_class(self) -> type[OpenRouterDeepseekModel]:
        return OpenRouterDeepseekModel

    def _max_price_for_level(self, level: int) -> dict[str, float]:
        """Per-level price ceiling, with explicit constructor args winning.

        Each bound is overridden independently so a caller can raise just the
        completion ceiling without restating the prompt one.
        """
        base = DEFAULT_MAX_PRICE_CHEAP if level == 1 else DEFAULT_MAX_PRICE_CAPABLE
        ceiling = dict(base)
        if self._max_price_prompt is not None:
            ceiling["prompt"] = self._max_price_prompt
        if self._max_price_completion is not None:
            ceiling["completion"] = self._max_price_completion
        return ceiling

    def _post_build_model(self, model: OpenRouterDeepseekModel, level: int) -> None:
        """Apply routing + (for DeepSeek) reasoning policy based on capability *level*.

        - Provider routing (order / allow_fallbacks / max_price / ignore) is
          model-agnostic — always applied.
        - Reasoning policy is DeepSeek-specific — only set for ``deepseek/`` models.
        - ``level == 1`` → reasoning disabled (cheap tier), cheap-tier ceiling
        - ``level != 1`` → reasoning at max effort (capable tier), capable ceiling
        - ``level == 0`` → sentinel for direct ``new_model()`` calls;
          applies capable-tier policy as a safe default.
        """
        model_name = str(getattr(model, "model_name", "") or "")
        if model_name.startswith("deepseek/"):
            if level == 1:
                # Cheap tier — verdict/generation work, no chain-of-thought.
                model.reasoning_setting = {"enabled": False}
            else:
                # Capable tier (or unknown level) — reasoning at max effort.
                model.reasoning_setting = {"effort": "xhigh"}
        if self._ignore_providers is not None:
            ignore = list(self._ignore_providers)
        elif level == 1:
            ignore = []
        else:
            ignore = list(DEFAULT_IGNORE_CAPABLE)
        model.provider_routing = build_provider_routing(
            preferred_provider=self._preferred_provider,
            allow_fallbacks=self._allow_fallbacks,
            max_price=self._max_price_for_level(level),
            ignore=ignore,
        )

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
    _PIN_MODEL_PREFIX,
    _PREFERRED_PROVIDER,
    DEFAULT_IGNORE_CAPABLE,
    DEFAULT_MAX_PRICE_CAPABLE,
    DEFAULT_MAX_PRICE_CHEAP,
    OpenRouterDeepseekModel,
    build_provider_routing,
)
from .provider import OpenRouterProvider

#: Sentinel for "``preferred_provider`` not passed": resolved per model at
#: build time — DeepSeek for ``deepseek/*`` models, no preference otherwise.
#: A bare ``"DeepSeek"`` default would pin a non-DeepSeek model to a provider
#: that does not serve it, while ``None`` would lose the DeepSeek default.
PREFERRED_PROVIDER_PER_MODEL: str = "<per-model default>"


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

    The routing knobs (``preferred_provider``, ``max_price_*``,
    ``ignore_providers``, ``allow_fallbacks``) apply to EVERY model served
    through this provider — explicit values are always honoured, and only
    the *defaults* differ by model family (DeepSeek-derived ceilings and
    ignore list for ``deepseek/*``, no routing constraints otherwise). The
    reasoning policy and the ``reasoning_content`` round-trip stay
    DeepSeek-specific.
    """

    def __init__(
        self,
        *,
        preferred_provider: str | None = PREFERRED_PROVIDER_PER_MODEL,
        allow_fallbacks: bool = True,
        max_price_prompt: float | None = None,
        max_price_completion: float | None = None,
        ignore_providers: list[str] | None = None,
        **kwargs: object,
    ) -> None:
        """Configure auth (see :class:`OpenRouterProvider`) plus routing policy.

        Args:
            preferred_provider: Upstream tried first. ``None`` lets OpenRouter
                choose freely. When omitted, ``deepseek/*`` models prefer
                DeepSeek and every other model expresses no preference.
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

    def _explicit_max_price(self) -> dict[str, float] | None:
        """Ceiling built ONLY from constructor args — no per-level defaults.

        Used for non-DeepSeek models, whose sticker prices the DeepSeek-derived
        defaults know nothing about (a too-low ceiling fails every request
        with ``404 No endpoints found``).
        """
        ceiling: dict[str, float] = {}
        if self._max_price_prompt is not None:
            ceiling["prompt"] = self._max_price_prompt
        if self._max_price_completion is not None:
            ceiling["completion"] = self._max_price_completion
        return ceiling or None

    def _post_build_model(self, model: OpenRouterDeepseekModel, level: int) -> None:
        """Apply routing + (for DeepSeek) reasoning policy based on capability *level*.

        Provider routing (order / allow_fallbacks / max_price / ignore) is
        applied to EVERY model: explicit constructor values — which is how a
        tier's ``provider_kwargs`` arrive — are always honoured. Only the
        defaults are model-family specific:

        - ``deepseek/`` models: DeepSeek preferred, per-level price ceiling,
          capable-tier ignore list; plus the DeepSeek-only reasoning policy
          (``level == 1`` → disabled, otherwise ``xhigh``; ``level == 0`` is
          the sentinel for direct ``new_model()`` calls and takes the capable
          policy as a safe default).
        - every other model: no preference, no ceiling, no ignore list unless
          given explicitly, so the request is priced by the model's own
          providers.

        Dropping the explicit knobs for non-DeepSeek models (the behaviour
        until 2026-08-28) silently discarded the level-2 defaults declared in
        :data:`~robotsix_llmio.config.tier.LEVEL2_DEFAULT`: OpenRouter then
        routed ``xiaomi/mimo-v2.5-pro`` freely and landed on providers whose
        cache-read rate is 20-45x Xiaomi's, multiplying the fleet's real
        level-2 cost several-fold on a cache-dominated workload.
        """
        model_name = str(getattr(model, "model_name", "") or "")
        is_deepseek = model_name.startswith(_PIN_MODEL_PREFIX)

        if is_deepseek:
            if level == 1:
                # Cheap tier — verdict/generation work, no chain-of-thought.
                model.reasoning_setting = {"enabled": False}
            else:
                # Capable tier (or unknown level) — reasoning at max effort.
                model.reasoning_setting = {"effort": "xhigh"}

        preferred = self._preferred_provider
        if preferred == PREFERRED_PROVIDER_PER_MODEL:
            preferred = _PREFERRED_PROVIDER if is_deepseek else None

        if self._ignore_providers is not None:
            ignore = list(self._ignore_providers)
        elif is_deepseek and level != 1:
            ignore = list(DEFAULT_IGNORE_CAPABLE)
        else:
            ignore = []

        max_price = (
            self._max_price_for_level(level)
            if is_deepseek
            else self._explicit_max_price()
        )

        model.provider_routing = build_provider_routing(
            preferred_provider=preferred,
            allow_fallbacks=self._allow_fallbacks,
            max_price=max_price,
            ignore=ignore,
        )

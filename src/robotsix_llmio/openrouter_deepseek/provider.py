"""Derived DeepSeek-on-OpenRouter provider — the layer consumers plug in.

Bakes the tier→model map and the tier→reasoning policy. The only consumer
knob is ``api_key`` (or ``OPENROUTER_API_KEY``); per agent the only choice is
the :class:`~robotsix_llmio.core.Tier`.
"""

from __future__ import annotations

from ..core.provider import Tier
from ..openrouter.provider import OpenRouterProvider
from .model import OpenRouterDeepseekModel

# Baked models — choosing this provider IS choosing these.
_DEFAULT_MODEL = "deepseek/deepseek-v4-pro"
_CHEAP_MODEL = "deepseek/deepseek-v4-flash"


class OpenRouterDeepseekProvider(OpenRouterProvider):
    """OpenRouter pinned to DeepSeek, with capable/cheap tiers."""

    def _tier_models(self) -> dict[Tier, str]:
        return {Tier.DEFAULT: _DEFAULT_MODEL, Tier.CHEAP: _CHEAP_MODEL}

    def _model_class(self) -> type[OpenRouterDeepseekModel]:
        return OpenRouterDeepseekModel

    def _post_build_model(self, model: OpenRouterDeepseekModel, tier: Tier) -> None:
        if tier == Tier.CHEAP:
            # Verdict/generation work — no chain-of-thought.
            model.reasoning_setting = {"enabled": False}
        else:
            # Capable tier — reasoning at max effort.
            model.reasoning_setting = {"effort": "xhigh"}

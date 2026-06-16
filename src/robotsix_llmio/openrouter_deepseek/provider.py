"""Derived DeepSeek-on-OpenRouter provider — the layer consumers plug in.

Model names are resolved from :class:`~robotsix_llmio.config.tier.TierConfig`
(via :meth:`~robotsix_llmio.core.provider.LLMProvider.build_agent`) or passed
directly to :meth:`new_model`.  The deprecated ``tier=`` path maps to these
model names via a minimal internal compat dict.
"""

from __future__ import annotations

from typing import ClassVar

from ..core.provider import Tier
from ..openrouter.provider import OpenRouterProvider
from .model import OpenRouterDeepseekModel


class OpenRouterDeepseekProvider(OpenRouterProvider):
    """OpenRouter pinned to DeepSeek, with per-level reasoning policy."""

    _tier_compat: ClassVar[dict[Tier, str]] = {
        Tier.DEFAULT: "deepseek/deepseek-v4-pro",
        Tier.CHEAP: "deepseek/deepseek-v4-flash",
    }

    def _model_class(self) -> type[OpenRouterDeepseekModel]:
        return OpenRouterDeepseekModel

    def _post_build_model(self, model: OpenRouterDeepseekModel, level: int) -> None:
        """Apply reasoning policy based on capability *level*.

        - ``level == 1`` → reasoning disabled (cheap tier)
        - ``level != 1`` → reasoning at max effort (capable tier)
        - ``level == 0`` → sentinel for direct ``new_model()`` calls;
          applies capable-tier policy as a safe default.
        """
        if level == 1:
            # Cheap tier — verdict/generation work, no chain-of-thought.
            model.reasoning_setting = {"enabled": False}
        else:
            # Capable tier (or unknown level) — reasoning at max effort.
            model.reasoning_setting = {"effort": "xhigh"}

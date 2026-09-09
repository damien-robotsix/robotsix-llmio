"""Direct-API DeepSeek transport layer.

Talks to ``https://api.deepseek.com`` directly with a DeepSeek-issued key,
reusing the DeepSeek-side behaviours (per-level thinking policy,
``reasoning_content`` round-trip) while dropping OpenRouter-only machinery
(provider-routing kwargs, ``usage.include``, server ``usage.cost``). Cost is
computed client-side from configurable sticker prices.

The model/provider (which import pydantic-ai and thus opentelemetry) are loaded
lazily via PEP 562 ``__getattr__`` so importing this package stays cheap.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from robotsix_llmio.exceptions import RobotsixLLMIOError


class DeepseekAPIError(RobotsixLLMIOError):
    """Error from the direct DeepSeek client (HTTP, auth, malformed response)."""


# Static re-declaration of every lazily-exported name (see ``__getattr__``
# below). These imports run ONLY under static analysis (``TYPE_CHECKING`` is
# False at runtime), so they add no import-time cost and preserve the PEP 562
# lazy-loading behaviour — but they let type checkers, IDEs, and CodeQL see
# each ``__all__`` entry as a defined module global. Without this, CodeQL's
# ``py/undefined-export`` query flags every ``__all__`` name as "exported but
# not defined" (it cannot model PEP 562 dynamic exports), failing the
# code-scanning check on any PR that adds a new export. Keep this block in
# sync with ``__all__`` and ``__getattr__``.
if TYPE_CHECKING:
    from .model import DeepseekModel
    from .pricing import (
        DeepseekPricing,
        ModelPrices,
        compute_cost,
        record_deepseek_cost,
    )
    from .provider import DeepseekProvider

__all__ = [
    "DeepseekAPIError",
    "DeepseekModel",
    "DeepseekPricing",
    "DeepseekProvider",
    "ModelPrices",
    "compute_cost",
    "record_deepseek_cost",
]


def __getattr__(name: str) -> Any:  # PEP 562 — lazy heavy imports
    if name == "DeepseekProvider":
        from .provider import DeepseekProvider

        return DeepseekProvider
    if name == "DeepseekModel":
        from .model import DeepseekModel

        return DeepseekModel
    if name in (
        "DeepseekPricing",
        "ModelPrices",
        "compute_cost",
        "record_deepseek_cost",
    ):
        from . import pricing

        return getattr(pricing, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

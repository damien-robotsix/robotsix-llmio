"""Direct-API Anthropic transport layer.

Talks to the first-party Anthropic Messages API (``https://api.anthropic.com``)
with an ``ANTHROPIC_API_KEY`` — a sibling of the Claude Agent SDK transport
(``claude_sdk``, which consumes the subscription seat via the ``claude`` CLI)
that spends first-party API credit instead. Built on pydantic-ai's native
``AnthropicModel``; cost is computed client-side from configurable sticker
prices (the API returns token counts, including cache read/write, but no cost).

The model/provider (which import pydantic-ai and the ``anthropic`` SDK) are
loaded lazily via PEP 562 ``__getattr__`` so importing this package stays cheap
and does not require the ``anthropic`` extra.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from robotsix_llmio.exceptions import RobotsixLLMIOError


class AnthropicAPIError(RobotsixLLMIOError):
    """Error from the direct Anthropic client (auth, configuration)."""


# Static re-declaration of every lazily-exported name (see ``__getattr__``
# below). These imports run ONLY under static analysis (``TYPE_CHECKING`` is
# False at runtime), so they add no import-time cost and preserve the PEP 562
# lazy-loading behaviour — but they let type checkers, IDEs, and CodeQL see
# each ``__all__`` entry as a defined module global. Keep this block in sync
# with ``__all__`` and ``__getattr__``.
if TYPE_CHECKING:
    from .model import AnthropicModel
    from .pricing import (
        AnthropicPricing,
        ModelPrices,
        compute_cost,
        record_anthropic_cost,
    )
    from .provider import AnthropicProvider

__all__ = [
    "AnthropicAPIError",
    "AnthropicModel",
    "AnthropicPricing",
    "AnthropicProvider",
    "ModelPrices",
    "compute_cost",
    "record_anthropic_cost",
]


def __getattr__(name: str) -> Any:  # PEP 562 — lazy heavy imports
    if name == "AnthropicProvider":
        from .provider import AnthropicProvider

        return AnthropicProvider
    if name == "AnthropicModel":
        from .model import AnthropicModel

        return AnthropicModel
    if name in (
        "AnthropicPricing",
        "ModelPrices",
        "compute_cost",
        "record_anthropic_cost",
    ):
        from . import pricing

        return getattr(pricing, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

"""Direct-API Anthropic model — pydantic-ai ``AnthropicModel`` + client-side cost.

The first-party Messages API returns token counts (including cache read/write)
but no dollar cost, so every completed request — non-streamed or streamed —
records a client-side cost onto the active span via
:func:`~robotsix_llmio.anthropic.pricing.record_anthropic_cost`.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Any

from pydantic_ai.models.anthropic import AnthropicModel as _PydAnthropicModel

from .pricing import AnthropicPricing, record_anthropic_cost


class AnthropicModel(_PydAnthropicModel):
    """pydantic-ai ``AnthropicModel`` that records cost client-side.

    The provider stamps :attr:`pricing` after construction; the default is used
    for direct construction.
    """

    #: Per-model sticker prices used to compute cost client-side.
    pricing: AnthropicPricing = AnthropicPricing()

    def _record_cost(self, response_usage: Any, model_id: str | None) -> None:
        record_anthropic_cost(response_usage, model_id or self.model_name, self.pricing)

    async def request(self, *args: Any, **kwargs: Any) -> Any:
        response = await super().request(*args, **kwargs)
        self._record_cost(response.usage, response.model_name)
        return response

    @asynccontextmanager
    async def request_stream(self, *args: Any, **kwargs: Any) -> AsyncIterator[Any]:
        async with super().request_stream(*args, **kwargs) as stream:
            try:
                yield stream
            finally:
                # Usage is cumulative over the stream; record once it is done
                # (or abandoned) while the request span is still active.
                self._record_cost(stream.usage, stream.model_name)

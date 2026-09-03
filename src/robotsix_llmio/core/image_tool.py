"""Image-question tool — lets text-only models interrogate attached images.

The transport rule: Anthropic (Claude SDK) models read images NATIVELY —
``ClaudeSDKProvider.build_agent(images=...)`` feeds them into the existing
native image-block flow and never touches this module. The baked OpenRouter
models (DeepSeek) are text-only — image input dies with 404 "No endpoints
found that support image input" — so for them the consumer passes the raw
images to :meth:`~robotsix_llmio.core.provider.LLMProvider.build_agent` via
``images=`` and llmio hands the agent an ``ask_image`` tool: the agent asks
natural-language questions about an attached image and gets a text answer,
produced by the :attr:`~robotsix_llmio.config.tier.TierConfig.vision`
binding (baked: ``openrouter-deepseek/deepseek-v4-flash-vision-exp``).

The tool is **async** deliberately: pydantic-ai awaits async tools natively,
and the Claude SDK MCP bridge invokes tools inside its running event loop —
a blocking sync tool would stall the loop, and a ``run_sync``-style call
inside it would raise.

Failures never raise into the agent loop: an out-of-range index, a missing
API key, or a vision-model error all come back as a plain explanatory string
the model can read and act on (and a warning is logged for the operator).
"""

from __future__ import annotations

import logging
from collections.abc import Callable, Coroutine, Sequence
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from robotsix_llmio.config.tier import TierConfig

log = logging.getLogger("robotsix_llmio.image_tool")

#: Hard cap on an ask_image answer — the caller's context is the scarce
#: resource, and a vision model rambling about a screenshot should not
#: crowd it out.
_MAX_ANSWER_CHARS = 4000

#: Appended to the system prompt by ``build_agent`` when images are attached.
IMAGE_NOTE_TEMPLATE = (
    "\n\n{n} image(s) are attached to this conversation. You cannot see them "
    "directly; use the ask_image tool (image_index 0..{max_index}) to ask "
    "questions about their content."
)


def build_image_question_tool(
    images: Sequence[tuple[str, bytes]],
    *,
    tier_config: TierConfig | None = None,
    api_key: str | None = None,
) -> Callable[[int, str], Coroutine[Any, Any, str]]:
    """Build the ``ask_image`` tool over *images*.

    Args:
        images: The attached images as ``(media_type, data)`` pairs, e.g.
            ``[("image/png", b"...")]``. Captured immutably at build time.
        tier_config: Supplies the :attr:`~robotsix_llmio.config.tier.TierConfig.vision`
            binding that answers the questions. ``None`` uses the baked
            defaults.
        api_key: API key for the vision provider (OpenRouter). ``None``
            falls back to the provider's own resolution
            (``OPENROUTER_API_KEY``).

    Returns:
        An async ``ask_image(image_index, question) -> str`` function ready
        to be passed as a pydantic-ai tool or through the Claude SDK MCP
        bridge.

    """
    frozen: tuple[tuple[str, bytes], ...] = tuple(
        (media_type, data) for media_type, data in images
    )

    if tier_config is None:
        from robotsix_llmio.config.tier import TierConfig as _TierConfig

        tier_config = _TierConfig()
    vision_tlc = tier_config.vision

    async def ask_image(image_index: int, question: str) -> str:
        """Ask a question about an attached image and get a text answer.

        Args:
            image_index: Which attached image to inspect, 0-based.
            question: A specific natural-language question about the image —
                e.g. "What error message is shown?" or "Describe the layout
                and any text in this screenshot."

        Returns:
            The answer as plain text (or an error description when the image
            cannot be inspected).

        """
        if not isinstance(image_index, int) or not 0 <= image_index < len(frozen):
            return (
                f"ask_image error: image_index {image_index!r} is out of range — "
                f"{len(frozen)} image(s) are attached (valid indices: "
                f"0..{len(frozen) - 1})."
            )

        media_type, data = frozen[image_index]
        try:
            answer = await _ask_vision_model(
                vision_tlc, media_type, data, question, api_key=api_key
            )
        except Exception as exc:  # never raise into the agent loop
            log.warning(
                "ask_image: vision call failed (%s): %s", type(exc).__name__, exc
            )
            return (
                f"ask_image error: the vision model could not be reached "
                f"({type(exc).__name__}: {exc}). Answer from the textual "
                f"context instead, and say the image could not be inspected."
            )

        if len(answer) > _MAX_ANSWER_CHARS:
            answer = answer[:_MAX_ANSWER_CHARS] + " …[truncated]"
        return answer

    return ask_image


def _augment_with_image_tool(
    system_prompt: str,
    tools: list[Any] | None,
    images: Sequence[tuple[str, bytes]],
    tier_config: TierConfig | None,
    api_key: str | None,
) -> tuple[str, list[Any]]:
    """``build_agent`` seam for TEXT-ONLY transports: append the ask_image
    tool and the system-prompt note. Called by the generic
    ``LLMProvider.build_agent``; the Claude SDK override serves images
    natively instead and never reaches this."""
    tool = build_image_question_tool(images, tier_config=tier_config, api_key=api_key)
    note = IMAGE_NOTE_TEMPLATE.format(n=len(images), max_index=len(images) - 1)
    return system_prompt + note, [*(tools or []), tool]


async def _ask_vision_model(
    vision_tlc: Any,
    media_type: str,
    data: bytes,
    question: str,
    *,
    api_key: str | None,
) -> str:
    """One-shot question to the vision binding; returns the text answer.

    Builds a fresh provider + agent per call (an image question is rare and
    the httpx2 client must not outlive it), runs under the provider's own
    transient retry, and closes the handle deterministically. Cost stamps
    onto the active OTel span exactly like every other OpenRouter call —
    under the tool span on the Claude SDK bridge, under the agent-run span
    on the pydantic-ai path.
    """
    from pydantic_ai.messages import BinaryContent

    from .factory import get_provider_for_identifier
    from .retry import acall_with_retry

    kwargs: dict[str, Any] = dict(vision_tlc.provider_kwargs)
    if api_key is not None:
        kwargs["api_key"] = api_key
    if vision_tlc.max_tokens is not None:
        kwargs.setdefault("max_tokens", vision_tlc.max_tokens)

    provider = get_provider_for_identifier(vision_tlc.model, **kwargs)
    handle = provider.build_agent(
        system_prompt=(
            "You answer questions about a single attached image, precisely "
            "and concisely. Transcribe any text relevant to the question "
            "verbatim."
        ),
        output_type=str,
        name="ask_image",
        model=vision_tlc.model_name,
    )
    try:

        async def _run() -> Any:
            return await handle.run(
                [question, BinaryContent(data=data, media_type=media_type)]
            )

        result = await acall_with_retry(
            _run, what="ask_image", is_transient_fn=provider._is_transient
        )
        return str(result.output)
    finally:
        handle.close()

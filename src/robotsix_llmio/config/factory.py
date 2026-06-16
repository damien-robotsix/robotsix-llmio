"""Consumer-facing factory — create a provider from *transport* + *model_level*.

This is the single entry-point consumers use to obtain a working model/agent.
They set ``transport`` and ``model_level`` in their config and call
:func:`create_model` — no direct provider-class or ``claude_agent_sdk`` import
is ever needed.

Built on top of :func:`~robotsix_llmio.core.factory.get_provider` and the
:class:`~robotsix_llmio.config.tier.TierConfig` schema.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from ..core.factory import get_provider

if TYPE_CHECKING:
    from ..core.provider import LLMProvider

from .transport import MODEL_LEVEL_TO_TIER, TRANSPORT_ALIASES


def create_model(
    *,
    transport: str,
    model_level: int,
    **provider_kwargs: Any,
) -> LLMProvider:
    """Create a provider instance for the given *transport* and *model_level*.

    Parameters
    ----------
    transport:
        Consumer-facing transport alias — one of ``"claude-sdk"`` or
        ``"openrouter[deepseek]"``.  Mapped to a provider registry name via
        :data:`~.transport.TRANSPORT_ALIASES`.
    model_level:
        Model strength level (1, 2, or 3).  Level 1 selects the cheap/fast
        model; levels 2 and 3 select the capable/default model.  Use
        :data:`~.transport.MODEL_LEVEL_TO_TIER` to obtain the corresponding
        :class:`~robotsix_llmio.core.provider.Tier` when calling
        :meth:`~robotsix_llmio.core.provider.LLMProvider.build_agent` or
        :meth:`~robotsix_llmio.core.provider.LLMProvider.new_model`.
    **provider_kwargs:
        Forwarded to the provider constructor (e.g. ``api_key=...`` for the
        OpenRouter provider, ``default_model=...`` for the Claude SDK
        provider).

    Returns
    -------
    LLMProvider
        A fully-instantiated provider ready for :meth:`~LLMProvider.build_agent`
        or :meth:`~LLMProvider.new_model` calls.

    Raises
    ------
    ValueError
        If *transport* is not a recognised alias.
    ImportError
        If the provider's optional extra is not installed.

    Example
    -------
    .. code-block:: python

        from robotsix_llmio.config import create_model, MODEL_LEVEL_TO_TIER

        provider = create_model(transport="claude-sdk", model_level=3)
        agent = provider.build_agent(
            tier=MODEL_LEVEL_TO_TIER[3],
            system_prompt="You are a helpful assistant.",
        )
    """
    if model_level not in MODEL_LEVEL_TO_TIER:
        valid = ", ".join(str(k) for k in sorted(MODEL_LEVEL_TO_TIER))
        raise ValueError(
            f"model_level must be one of {valid}; got {model_level!r}"
        )

    try:
        provider_name = TRANSPORT_ALIASES[transport]
    except KeyError as exc:
        known = ", ".join(sorted(TRANSPORT_ALIASES))
        raise ValueError(
            f"Unknown transport {transport!r}. Known transports: {known}."
        ) from exc

    return get_provider(provider=provider_name, **provider_kwargs)

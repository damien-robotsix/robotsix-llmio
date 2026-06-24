"""Provider-agnostic base: the ``LLMProvider`` ABC."""

from __future__ import annotations

import time
from abc import ABC, abstractmethod
from collections.abc import Callable
from typing import TYPE_CHECKING, Any, TypeVar

from . import retry as _retry
from .agent import AgentHandle
from .agent import build_agent as _build_agent

if TYPE_CHECKING:
    from robotsix_llmio.config.tier import TierConfig

T = TypeVar("T")


def _resolve_output_type(output_type: Any, level: int) -> Any:
    """Resolve *output_type* for the given capability *level*.

    At **level >= 2** the model runs in extended-thinking (reasoning) mode.
    pydantic-ai's default for a raw structured type (a pydantic
    ``BaseModel`` subclass, dataclass, etc.) is :class:`~pydantic_ai.ToolOutput`
    — which forces ``tool_choice`` in the API call. DeepSeek's thinking mode
    rejects a forced ``tool_choice``, so a raw structured type must be
    converted to :class:`~pydantic_ai.PromptedOutput` (prompt-based JSON
    output) to coexist with thinking.

    Rules
    -----

    * ``level < 2`` — return *output_type* unchanged.
    * ``output_type is str`` — return ``str`` unchanged (plain-text mode).
    * *output_type* already an explicit output-mode marker instance
      (:class:`~pydantic_ai.PromptedOutput`,
       :class:`~pydantic_ai.ToolOutput`, or
       :class:`~pydantic_ai.NativeOutput`) — return unchanged; never
      double-wrap.
    * *output_type* is a ``list`` / ``tuple`` that **contains** any
      explicit marker instance — return unchanged; the caller has
      expressed explicit intent.
    * Otherwise (a raw structured type, or a ``list`` / ``tuple`` / union of
      plain types) **and** ``level >= 2`` — return
      ``PromptedOutput(output_type)``.
    """
    if level < 2:
        return output_type

    if output_type is str:
        return output_type

    from pydantic_ai import NativeOutput, PromptedOutput, ToolOutput

    _MARKERS = (PromptedOutput, ToolOutput, NativeOutput)

    if isinstance(output_type, _MARKERS):
        return output_type

    if isinstance(output_type, (list, tuple)) and any(
        isinstance(entry, _MARKERS) for entry in output_type
    ):
        return output_type

    # Not str, not an explicit marker, not a container-of-markers, and
    # level >= 2 — wrap in PromptedOutput to avoid forced tool_choice.
    return PromptedOutput(output_type)


def _resolve_model_name(
    tier_config: TierConfig | None,
    level: int,
) -> str:
    """Resolve a concrete model name from *tier_config* for the given *level*.

    When *tier_config* is ``None``, a default :class:`TierConfig` is built
    from the baked module-level defaults
    (:data:`~robotsix_llmio.config.tier.LEVEL1_DEFAULT`,
    :data:`~robotsix_llmio.config.tier.LEVEL2_DEFAULT`,
    :data:`~robotsix_llmio.config.tier.LEVEL3_DEFAULT`).
    """
    if tier_config is None:
        from robotsix_llmio.config.tier import (
            LEVEL1_DEFAULT,
            LEVEL2_DEFAULT,
            LEVEL3_DEFAULT,
            TierConfig,
        )

        tier_config = TierConfig(
            level1=LEVEL1_DEFAULT,
            level2=LEVEL2_DEFAULT,
            level3=LEVEL3_DEFAULT,
        )
    return tier_config.for_level(level).model_name


class LLMProvider(ABC):
    """Base for every provider. A derived provider implements :meth:`new_model`
    (and optionally :meth:`_is_transient`); the generic ``build_agent`` /
    ``call_with_retry`` are inherited."""

    @abstractmethod
    def new_model(
        self,
        *,
        model: str | None = None,
        level: int = 0,
    ) -> tuple[Any, Any]:
        """Return ``(model, http_client)`` — a fully configured pydantic-ai
        model (provider/auth/cost/quirks baked in) plus the http client to
        close when done.

        Parameters
        ----------
        model:
            The concrete model name (e.g. ``"deepseek/deepseek-v4-flash"``).
        level:
            Capability level (1, 2, or 3) for per-level policy hooks.
            ``0`` is the sentinel for "unknown / direct ``new_model()``
            call" — providers should apply a safe default.
        """
        raise NotImplementedError

    def _is_transient(self, exc: BaseException) -> bool:
        """Transient predicate for ``call_with_retry``. Override to widen with
        provider-specific signatures."""
        return _retry.is_transient(exc)

    def build_agent(
        self,
        *,
        level: int = 1,
        tier_config: TierConfig | None = None,
        model: str | None = None,
        system_prompt: str,
        tools: list[Any] | None = None,
        output_type: Any = str,
        name: str | None = None,
        retries: int = 2,
        builtin_tools: bool = True,
    ) -> AgentHandle:
        """Build a ready-to-run agent for the requested capability *level*.

        *builtin_tools* is honored only by the claude-sdk provider (whose SDK
        ships a built-in toolset); set it ``False`` to restrict that agent to
        only the explicitly provided tools. Other providers expose no built-in
        tools and accept it as a no-op.

        Parameters
        ----------
        level:
            Integer 1-3 selecting the capability tier:

            - ``1`` — cheap, fast, repetitive tasks
            - ``2`` — intermediate, e.g. implementing code
            - ``3`` — high-level planning / refine

        tier_config:
            When provided, resolution is::

                tlc = tier_config.for_level(level)
                new_model(model=tlc.model_name)

            When ``None``, a default :class:`~robotsix_llmio.config.tier.TierConfig`
            is built from the baked module-level defaults
            (:data:`~robotsix_llmio.config.tier.LEVEL1_DEFAULT`,
            :data:`~robotsix_llmio.config.tier.LEVEL2_DEFAULT`,
            :data:`~robotsix_llmio.config.tier.LEVEL3_DEFAULT`).

            Ignored when *model* is provided (see below).
        model:
            Optional explicit **bare** model name override (not a full
            provider-model identifier).  When provided, this is passed
            directly to :meth:`new_model`, bypassing *tier_config*
            resolution entirely.  The caller is responsible for providing
            a valid model name for the provider.  When ``None`` (default),
            the model is resolved from *tier_config* as usual.
        system_prompt:
            Final system prompt for the agent (domain concern).
        tools:
            Optional list of Python functions the agent may call.
        output_type:
            Expected return type; defaults to :class:`str`.  At ``level >= 2``
            a raw structured type (pydantic model, dataclass, etc.) is
            automatically wrapped in
            :class:`~pydantic_ai.PromptedOutput` to avoid the forced
            ``tool_choice`` / thinking-mode conflict.  Pass an explicit
            :class:`~pydantic_ai.ToolOutput`,
            :class:`~pydantic_ai.NativeOutput`, or
            :class:`~pydantic_ai.PromptedOutput` to opt out of the
            auto-wrap.
        name:
            Optional name for the agent (used in traces).
        retries:
            Maximum retry attempts for the underlying pydantic-ai agent
            (default ``2``).

        Returns
        -------
        AgentHandle
            A ready-to-run agent handle wrapping a pydantic-ai ``Agent``
            and its ``httpx`` client.  Call ``.close()`` when done.
        """
        resolved_output_type = _resolve_output_type(output_type, level)

        if model is not None:
            # Explicit override — bypass tier_config entirely.
            m, http_client = self.new_model(model=model, level=level)
            return _build_agent(
                m,
                http_client,
                system_prompt=system_prompt,
                tools=tools,
                output_type=resolved_output_type,
                name=name,
                retries=retries,
            )

        model_name = _resolve_model_name(tier_config, level)
        m, http_client = self.new_model(model=model_name, level=level)

        return _build_agent(
            m,
            http_client,
            system_prompt=system_prompt,
            tools=tools,
            output_type=resolved_output_type,
            name=name,
            retries=retries,
        )

    def call_with_retry(
        self,
        fn: Callable[[], T],
        *,
        what: str = "model call",
        sleep: Callable[[float], None] = time.sleep,
        fallback_fn: Callable[[], T] | None = None,
    ) -> T:
        """Run *fn* with bounded transient/rate-limit retry, using this
        provider's transient signatures."""
        return _retry.call_with_retry(
            fn,
            what=what,
            sleep=sleep,
            fallback_fn=fallback_fn,
            is_transient_fn=self._is_transient,
        )

"""Provider-agnostic base: the ``LLMProvider`` ABC."""

from __future__ import annotations

import time
from abc import ABC, abstractmethod
from collections.abc import Callable, Sequence
from typing import TYPE_CHECKING, Any, TypeVar

from . import retry as _retry
from .agent import AgentHandle
from .agent import build_agent as _build_agent

if TYPE_CHECKING:
    from robotsix_llmio.config.tier import TierConfig

T = TypeVar("T")


# Appended to the system prompt when PromptedOutput is active, to guard
# against models (notably deepseek-v4-flash) that occasionally hallucinate
# DSML / <invoke> tool-call markup instead of emitting raw JSON.  The
# instruction is deliberately terse — the primary JSON-schema instruction
# comes from pydantic-ai; this is a reinforcement.
_ANTI_DSML_INSTRUCTION = (
    "\n\nIMPORTANT: Output ONLY the raw JSON object. "
    "Do NOT wrap your response in any XML/DSML markup, "
    "tool-call tags like <DSML> or <invoke>, "
    "or any other formatting. "
    "The response must be pure JSON with no surrounding text."
)


def _is_prompted_output(output_type: Any) -> bool:
    """Return True if *output_type* will use pydantic-ai's PromptedOutput mode."""
    from pydantic_ai import PromptedOutput

    if isinstance(output_type, PromptedOutput):
        return True
    if isinstance(output_type, (list, tuple)):
        return any(isinstance(o, PromptedOutput) for o in output_type)
    return False


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

    from pydantic_ai import PromptedOutput

    from robotsix_llmio.core._output_markers import _is_output_type_marked

    if _is_output_type_marked(output_type):
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
    via the no-arg constructor, which produces independent deep copies of
    the module-level baked defaults. Resolution honours the provider slot
    the failover tracker currently designates as active.
    """
    if tier_config is None:
        from robotsix_llmio.config.tier import TierConfig

        tier_config = TierConfig()
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
            The concrete model name (e.g. ``"deepseek/deepseek-v4-flash-latest"``).
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
        web_tools: bool = False,
        images: Sequence[tuple[str, bytes]] | None = None,
        vision_api_key: str | None = None,
    ) -> AgentHandle:
        """Build a ready-to-run agent for the requested capability *level*.

        *builtin_tools* is honored only by the claude-sdk provider (whose SDK
        ships a built-in toolset); set it ``False`` to restrict that agent to
        only the explicitly provided tools. Other providers expose no built-in
        tools and accept it as a no-op.

        Parameters
        ----------
        level:
            Integer 1-3 selecting the capability level:

            - ``1`` — cheap, frequent: monitors, classifiers, summaries
            - ``2`` — workhorse: implementing code, main assistant turns
            - ``3`` — frontier: hardest reasoning and long-horizon work

        tier_config:
            When provided, resolution is::

                tlc = tier_config.for_level(level)
                new_model(model=tlc.model_name)

            When ``None``, a default :class:`~robotsix_llmio.config.tier.TierConfig`
            is built via the no-arg constructor, which produces independent
            deep copies of the baked defaults.

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
        builtin_tools:
            When ``True`` (default), allows the provider to expose its
            built-in toolset (only honored by the claude-sdk provider).
            Set ``False`` to restrict the agent to only the explicitly
            provided *tools*.
        web_tools:
            Grants ``WebFetch`` / ``WebSearch`` even when *builtin_tools* is
            ``False``. They read nothing local and mutate nothing, so an agent
            restricted from the filesystem and shell can still look something
            up. Default ``False`` — an agent that never needs the web should
            not carry the capability. Also only honored by the claude-sdk
            provider.
        images:
            Attached images as ``(media_type, bytes)`` pairs. Transport-
            dependent: the Claude SDK provider overrides this to deliver
            them NATIVELY (Claude models read images). Here — the generic
            path serving OpenRouter's text-only models — the agent instead
            gets an ``ask_image`` tool answered by the tier config's
            ``vision`` binding, plus a system-prompt note pointing at it.
            Either way the caller's prompt stays TEXT-ONLY: do not embed
            ``BinaryContent`` when passing ``images=``.
        vision_api_key:
            API key for the vision binding's provider. Precedence: this
            argument, else the provider's own OpenRouter key (when this IS
            an OpenRouter-family provider), else the vision provider's
            environment fallback (``OPENROUTER_API_KEY``).

        Returns
        -------
        AgentHandle
            A ready-to-run agent handle wrapping a pydantic-ai ``Agent``
            and its ``httpx`` client.  Call ``.close()`` when done.

        """
        if images:
            from .image_tool import _augment_with_image_tool

            system_prompt, tools = _augment_with_image_tool(
                system_prompt,
                tools,
                images,
                tier_config,
                vision_api_key or getattr(self, "_api_key", None),
            )

        resolved_output_type = _resolve_output_type(output_type, level)

        # When using PromptedOutput, harden the system prompt against models
        # that hallucinate DSML/XML tool-call markup instead of JSON.
        if _is_prompted_output(resolved_output_type):
            system_prompt = system_prompt + _ANTI_DSML_INSTRUCTION

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

"""Provider-agnostic base: the ``Tier`` enum and the ``LLMProvider`` ABC."""

from __future__ import annotations

import time
import warnings
from abc import ABC, abstractmethod
from collections.abc import Callable
from enum import StrEnum
from typing import TYPE_CHECKING, Any, TypeVar

from . import retry as _retry
from .agent import AgentHandle
from .agent import build_agent as _build_agent

if TYPE_CHECKING:
    from robotsix_llmio.config.tier import TierConfig

T = TypeVar("T")


def _level_to_tier(level: int) -> Tier:
    """Map an integer level (1-3) to the legacy :class:`Tier` enum.

    **Deprecated** — prefer :meth:`TierConfig.for_level` which resolves
    directly to a :class:`~robotsix_llmio.config.tier.TierLevelConfig`
    without the two-tier round-trip.

    +----------+--------------------------------------+
    | ``level`` | :class:`Tier`                        |
    +==========+======================================+
    | 1        | :attr:`Tier.CHEAP`                   |
    +----------+--------------------------------------+
    | 2        | :attr:`Tier.DEFAULT`                 |
    +----------+--------------------------------------+
    | 3        | :attr:`Tier.DEFAULT` *(stop-gap)*    |
    +----------+--------------------------------------+

    Level 3 currently maps to ``Tier.DEFAULT``.  Full three-tier
    differentiation at the provider level is deferred to a follow-up;
    see the migration notes in the README.
    """
    if level == 1:
        return Tier.CHEAP
    if level in (2, 3):
        return Tier.DEFAULT
    raise ValueError(f"`level` must be 1, 2, or 3, got {level!r}")


class Tier(StrEnum):
    """**Deprecated** two-tier model selector — replaced by
    :class:`~robotsix_llmio.config.tier.TierConfig` and the integer *level*
    parameter on :meth:`LLMProvider.build_agent`.

    This enum remains for backward compatibility only.  Passing
    ``tier=Tier.CHEAP`` (or ``Tier.DEFAULT``) to ``build_agent()`` or
    ``new_model()`` still works but emits a :exc:`DeprecationWarning`.
    New code should use ``level=`` (1/2/3) and a ``TierConfig``.

    ========== =========== ============
    Member     Value       Replacement
    ========== =========== ============
    DEFAULT    ``default`` ``level=2``
    CHEAP      ``cheap``   ``level=1``
    ========== =========== ============
    """

    DEFAULT = "default"  # capable tier
    CHEAP = "cheap"  # fast/cheap tier


class LLMProvider(ABC):
    """Base for every provider. A derived provider implements :meth:`new_model`
    (and optionally :meth:`_is_transient`); the generic ``build_agent`` /
    ``call_with_retry`` are inherited."""

    @abstractmethod
    def new_model(
        self,
        *,
        model: str | None = None,
        tier: Tier | None = None,
        level: int = 0,
    ) -> tuple[Any, Any]:
        """Return ``(model, http_client)`` — a fully configured pydantic-ai
        model (provider/auth/cost/quirks baked in) plus the http client to
        close when done.

        Parameters
        ----------
        model:
            **Primary** — the concrete model name (e.g.
            ``"deepseek/deepseek-v4-flash"``).  When provided the model is
            constructed directly; *tier* is ignored.
        tier:
            **Deprecated** — use *model* instead.  When *model* is ``None``
            and *tier* is provided, the provider resolves via a minimal
            internal compat dict and emits a :exc:`DeprecationWarning`.
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
        tier: Tier | None = None,
        tier_config: TierConfig | None = None,
        system_prompt: str,
        tools: list[Any] | None = None,
        output_type: Any = str,
        name: str | None = None,
        retries: int = 2,
    ) -> AgentHandle:
        """Build a ready-to-run agent for the requested capability *level*.

        Parameters
        ----------
        level:
            Integer 1-3 selecting the capability tier:

            - ``1`` — cheap, fast, repetitive tasks
            - ``2`` — intermediate, e.g. implementing code
            - ``3`` — high-level planning / refine

        tier:
            **Deprecated** — use *level* and *tier_config* instead.
            Passing *tier* explicitly still works but emits a
            :exc:`DeprecationWarning`.

        tier_config:
            When provided, resolution is::

                tlc = tier_config.for_level(level)
                new_model(model=tlc.model)

            When ``None`` (backward-compat path), emits a
            :exc:`DeprecationWarning` and falls back to the legacy
            ``_level_to_tier(level)`` → ``new_model(tier=Tier.xxx)`` path.

        system_prompt:
            Final system prompt for the agent (domain concern).
        tools:
            Optional list of Python functions the agent may call.
        output_type:
            Expected return type; defaults to :class:`str`.
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
        # Emit deprecation warning for the legacy *tier* parameter
        # unconditionally when provided, even if *tier_config* is also set
        # (tier is superseded).
        if tier is not None:
            warnings.warn(
                "The `tier` parameter is deprecated. "
                "Use `level` and `tier_config` instead.",
                DeprecationWarning,
                stacklevel=2,
            )

        if tier_config is not None:
            # Primary path: resolve through TierConfig
            tlc = tier_config.for_level(level)
            model, http_client = self.new_model(model=tlc.model, level=level)
        else:
            # Legacy fallback path: resolve through Tier enum
            warnings.warn(
                "`tier_config` not provided — using legacy "
                "_level_to_tier() path.  Pass a `TierConfig` instance "
                "to `build_agent(tier_config=...)`.",
                DeprecationWarning,
                stacklevel=2,
            )
            resolved_tier = tier if tier is not None else _level_to_tier(level)
            model, http_client = self.new_model(tier=resolved_tier, level=level)

        return _build_agent(
            model,
            http_client,
            system_prompt=system_prompt,
            tools=tools,
            output_type=output_type,
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

"""Provider-agnostic base: the ``Tier`` enum and the ``LLMProvider`` ABC."""

from __future__ import annotations

import time
import warnings
from abc import ABC, abstractmethod
from collections.abc import Callable
from enum import StrEnum
from typing import Any, TypeVar

from . import retry as _retry
from .agent import AgentHandle
from .agent import build_agent as _build_agent

T = TypeVar("T")


def _level_to_tier(level: int) -> Tier:
    """Map an integer level (1-3) to the legacy :class:`Tier` enum.

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
    """Deprecated two-tier model selector.

    Prefer :class:`~robotsix_llmio.config.tier.TierLevel` (``level``
    parameter on :meth:`LLMProvider.build_agent`).

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
    def new_model(self, tier: Tier = Tier.DEFAULT) -> tuple[Any, Any]:
        """Return ``(model, http_client)`` for *tier* — a fully configured
        pydantic-ai model (provider/auth/cost/quirks baked in) plus the http
        client to close when done."""
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

            - ``1`` — cheap, fast, repetitive tasks (:attr:`Tier.CHEAP`)
            - ``2`` — intermediate, e.g. implementing code (:attr:`Tier.DEFAULT`)
            - ``3`` — high-level planning / refine (currently maps to
              :attr:`Tier.DEFAULT`; full three-tier differentiation is
              deferred to a follow-up).

        tier:
            **Deprecated** — use *level* instead.  Passing *tier* explicitly
            still works but emits a :exc:`DeprecationWarning`.  When both
            *level* (non-default) and *tier* are passed, *level* takes
            precedence.

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
        # Resolve level → legacy Tier
        if tier is None:
            resolved_tier = _level_to_tier(level)
        else:
            warnings.warn(
                "The `tier` parameter is deprecated. "
                "Use `level` instead (1 → Tier.CHEAP, 2 → Tier.DEFAULT, "
                "3 → Tier.DEFAULT [stop-gap]).",
                DeprecationWarning,
                stacklevel=2,
            )
            resolved_tier = _level_to_tier(level) if level != 1 else tier

        model, http_client = self.new_model(resolved_tier)
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

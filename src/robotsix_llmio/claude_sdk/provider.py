"""Claude Agent SDK provider — subscription-auth transport, one model per tier.

Sibling of the OpenRouter layer (both derive from :class:`core.LLMProvider`),
but it speaks to no HTTP endpoint: it drives the local ``claude`` CLI via the
Claude Agent SDK, so it needs **no API key** — only a logged-in ``claude``
(``claude login``) and Node.js on PATH.

Model names are resolved from :class:`~robotsix_llmio.config.tier.TierConfig`
(via :meth:`~robotsix_llmio.core.provider.LLMProvider.build_agent`) or passed
directly to :meth:`new_model`.
"""

from __future__ import annotations

import warnings
from pathlib import Path
from typing import TYPE_CHECKING, Any

from ..core.provider import LLMProvider, Tier, _level_to_tier
from ._tool_agent import _convert_tools, _SdkToolAgentHandle
from .transient import is_claude_sdk_transient

if TYPE_CHECKING:
    from robotsix_llmio.config.tier import TierConfig

# Minimal internal compat dict for the deprecated ``tier=`` path.
_TIER_COMPAT: dict[Tier, str] = {Tier.DEFAULT: "opus", Tier.CHEAP: "haiku"}


class ClaudeSDKProvider(LLMProvider):
    """Builds :class:`~robotsix_llmio.claude_sdk.model.ClaudeSDKModel` instances,
    one per model name, authenticated by your ``claude login`` subscription."""

    def __init__(self) -> None:
        # No constructor kwargs needed — model names are passed at
        # ``new_model()`` time via `model=` (or via the deprecated `tier=`
        # fallback).
        pass

    def new_model(
        self,
        *,
        model: str | None = None,
        tier: Tier | None = None,
        level: int = 0,
    ) -> tuple[Any, Any]:
        """Build a model, returning ``(model, http_client)``.

        The ``http_client`` is always ``None`` here: the ``claude`` CLI
        subprocess is the transport and there is no HTTP client to manage.

        Parameters
        ----------
        model:
            **Primary** — the concrete model name (e.g. ``"haiku"``,
            ``"opus"``).  When provided the model is constructed directly;
            *tier* is ignored.
        tier:
            **Deprecated** — use *model* instead.  When *model* is ``None``
            and *tier* is provided, resolves via a minimal internal compat
            dict and emits a :exc:`DeprecationWarning`.
        level:
            Capability level (unused for Claude SDK — no per-level policy
            is applied here).  Accepted for signature compatibility with
            the base class.
        """
        from .model import ClaudeSDKModel

        if model is not None:
            model_name = model
        elif tier is not None:
            warnings.warn(
                "The `tier` parameter on `new_model()` is deprecated. "
                "Pass `model=` with a concrete model name instead.",
                DeprecationWarning,
                stacklevel=2,
            )
            model_name = _TIER_COMPAT[tier]
        else:
            raise ValueError(
                "Either `model` or `tier` must be provided to `new_model()`."
            )

        # No http_client to manage — the CLI subprocess is the transport, and
        # the SDK tears it down per call. AgentHandle.close() tolerates None.
        return ClaudeSDKModel(model_name), None

    def _is_transient(self, exc: BaseException) -> bool:
        return is_claude_sdk_transient(exc)

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
        workspace_root: str | Path | None = None,
    ) -> Any:
        """Build a ready-to-run agent for the requested capability *level*.

        Mirrors :meth:`LLMProvider.build_agent`: prefer the integer *level*
        (1 → cheap, 2/3 → capable) with *tier_config*; the legacy *tier*
        parameter is deprecated but still honoured.

        When *tools* is non-empty, returns a :class:`_SdkToolAgentHandle` that
        drives the SDK tool loop directly — intermediate ``ToolCallPart``
        objects are not surfaced.  When *tools* is empty/``None``, delegates
        to the standard pydantic-ai ``Agent`` path (unchanged).

        *workspace_root* confines the agent's built-in file-mutating tools
        (``Write``/``Edit``/``MultiEdit``/``NotebookEdit``) to that directory:
        the SDK runs with ``cwd=workspace_root`` and a ``PreToolUse`` hook
        denies any edit whose target resolves outside it. Without it a
        tool-bearing agent can edit files anywhere the process reaches (e.g.
        the host app's own source) because ``permission_mode`` is
        ``bypassPermissions``. All built-in tools stay available — only
        out-of-scope *writes* are refused. Ignored on the no-tools path
        (no tools → nothing to confine)."""
        if not tools:
            return super().build_agent(
                level=level,
                tier=tier,
                tier_config=tier_config,
                system_prompt=system_prompt,
                tools=tools,
                output_type=output_type,
                name=name,
                retries=retries,
            )

        # Tool path: resolve model name from tier_config (primary) or
        # fall back to the legacy tier-based resolution.
        if tier is not None:
            warnings.warn(
                "The `tier` parameter is deprecated. "
                "Use `level` and `tier_config` instead.",
                DeprecationWarning,
                stacklevel=2,
            )

        if tier_config is not None:
            tlc = tier_config.for_level(level)
            sdk_model = tlc.model
        else:
            warnings.warn(
                "`tier_config` not provided — using legacy "
                "_level_to_tier() path.  Pass a `TierConfig` instance "
                "to `build_agent(tier_config=...)`.",
                DeprecationWarning,
                stacklevel=2,
            )
            resolved_tier = tier if tier is not None else _level_to_tier(level)
            sdk_model = _TIER_COMPAT[resolved_tier]

        allowed_tools, server = _convert_tools(tools)
        return _SdkToolAgentHandle(
            sdk_model=sdk_model,
            system_prompt=system_prompt,
            server=server,
            allowed_tools=allowed_tools,
            output_type=output_type,
            name=name,
            workspace_root=workspace_root,
        )

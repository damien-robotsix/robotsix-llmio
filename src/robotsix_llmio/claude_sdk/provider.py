"""Claude Agent SDK provider — subscription-auth transport, one model per tier.

Sibling of the OpenRouter layer (both derive from :class:`core.LLMProvider`),
but it speaks to no HTTP endpoint: it drives the local ``claude`` CLI via the
Claude Agent SDK, so it needs **no API key** — only a logged-in ``claude``
(``claude login``) and Node.js on PATH.

The only consumer knob is the :class:`~robotsix_llmio.core.Tier`; the tier→model
map is baked (overridable at construction for experimentation).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from ..core.provider import LLMProvider, Tier, _level_to_tier
from ._tool_agent import _convert_tools, _SdkToolAgentHandle
from .transient import is_claude_sdk_transient

# Baked tier→model map. Values are Claude Code model aliases passed straight to
# the SDK's ``model`` option (it resolves them to the latest concrete model).
_DEFAULT_MODEL = "opus"
_CHEAP_MODEL = "haiku"


class ClaudeSDKProvider(LLMProvider):
    """Builds :class:`~robotsix_llmio.claude_sdk.model.ClaudeSDKModel` instances,
    one per tier, authenticated by your ``claude login`` subscription."""

    def __init__(
        self,
        *,
        default_model: str = _DEFAULT_MODEL,
        cheap_model: str = _CHEAP_MODEL,
    ) -> None:
        self._models = {Tier.DEFAULT: default_model, Tier.CHEAP: cheap_model}

    def new_model(self, tier: Tier = Tier.DEFAULT) -> tuple[Any, Any]:
        """Build a model for *tier*, returning ``(model, http_client)``.

        The ``http_client`` is always ``None`` here: the ``claude`` CLI
        subprocess is the transport and there is no HTTP client to manage.
        """
        from .model import ClaudeSDKModel

        # No http_client to manage — the CLI subprocess is the transport, and
        # the SDK tears it down per call. AgentHandle.close() tolerates None.
        return ClaudeSDKModel(self._models[tier]), None

    def _is_transient(self, exc: BaseException) -> bool:
        return is_claude_sdk_transient(exc)

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
        workspace_root: str | Path | None = None,
    ) -> Any:
        """Build a ready-to-run agent for the requested capability *level*.

        Mirrors :meth:`LLMProvider.build_agent`: prefer the integer *level*
        (1 → cheap, 2/3 → capable); the legacy *tier* parameter is deprecated
        but still honoured. On the no-tools path the call is delegated to the
        base implementation (which emits the :exc:`DeprecationWarning` for an
        explicit *tier*).

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
                system_prompt=system_prompt,
                tools=tools,
                output_type=output_type,
                name=name,
                retries=retries,
            )

        # Tool path: resolve to a concrete tier silently (an explicit *tier*
        # is honoured without warning to keep the SDK tool loop quiet).
        resolved_tier = tier if tier is not None else _level_to_tier(level)
        sdk_model = self._models[resolved_tier]
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

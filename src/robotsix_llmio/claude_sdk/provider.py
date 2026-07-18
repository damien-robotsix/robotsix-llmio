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

from pathlib import Path
from typing import TYPE_CHECKING, Any

from ..core.provider import LLMProvider, _resolve_model_name
from ._tool_agent import _SdkToolAgentHandle
from ._tool_converter import _convert_tools
from .transient import is_claude_sdk_transient

if TYPE_CHECKING:
    from robotsix_llmio.config.tier import TierConfig


class ClaudeSDKProvider(LLMProvider):
    """Builds :class:`~robotsix_llmio.claude_sdk.model.ClaudeSDKModel` instances,
    one per model name, authenticated by your ``claude login`` subscription."""

    def __init__(self) -> None:
        # No constructor kwargs needed — model names are passed at
        # ``new_model()`` time via `model=`.
        pass

    def new_model(
        self,
        *,
        model: str | None = None,
        level: int = 0,
    ) -> tuple[Any, Any]:
        """Build a model, returning ``(model, http_client)``.

        The ``http_client`` is always ``None`` here: the ``claude`` CLI
        subprocess is the transport and there is no HTTP client to manage.

        Args:
            model: The concrete model name (e.g. ``"haiku"``, ``"opus"``).
            level: Capability level (unused for Claude SDK — no per-level
                policy is applied here).  Accepted for signature
                compatibility with the base class.

        """
        from .model import ClaudeSDKModel

        if model is None:
            raise ValueError("`model` must be provided to `new_model()`.")

        # No http_client to manage — the CLI subprocess is the transport, and
        # the SDK tears it down per call. AgentHandle.close() tolerates None.
        return ClaudeSDKModel(model), None

    def _is_transient(self, exc: BaseException) -> bool:
        return is_claude_sdk_transient(exc)

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
        workspace_root: str | Path | None = None,
        builtin_tools: bool = True,
    ) -> Any:
        """Build a ready-to-run agent for the requested capability *level*.

        Uses the integer *level* (1 → cheap, 2/3 → capable, 4 → frontier)
        with *tier_config*.  When *tier_config* is ``None``, a default is
        built from baked module-level defaults.

        When *tools* is non-empty, returns a :class:`_SdkToolAgentHandle` that
        drives the SDK tool loop directly — intermediate ``ToolCallPart``
        objects are not surfaced.  When *tools* is empty/``None``, delegates
        to the standard pydantic-ai ``Agent`` path (unchanged).

        Args:
            model: Optional explicit model name override (e.g.
                ``"haiku"``).  When provided, this becomes the
                ``sdk_model`` directly, bypassing *tier_config*.  When
                ``None``, the model is resolved from *tier_config* as
                usual.

            *workspace_root* confines the agent's built-in file-mutating
                tools to that directory: the SDK runs with
                ``cwd=workspace_root`` and a ``PreToolUse`` hook denies
                any write whose target resolves outside it. Without it a
                tool-bearing agent can edit files anywhere the process
                reaches (e.g. the host app's own source) because
                ``permission_mode`` is ``bypassPermissions``. All
                built-in tools stay available — only out-of-scope file
                mutations are refused, from both the built-in edit tools
                (``Write``/``Edit``/``MultiEdit``/``NotebookEdit``) and
                ``Bash`` commands whose arguments reference absolute
                paths outside the workspace. Ignored on the no-tools
                path (no tools → nothing to confine).

        """
        if not tools:
            # _resolve_output_type's level < 2 early-return is a DeepSeek-specific
            # rule and does not apply to ClaudeSDKModel, which requires PromptedOutput
            # at ALL levels for non-str structured output.
            if output_type is not str:
                from pydantic_ai import PromptedOutput

                from robotsix_llmio.core._output_markers import _is_output_type_marked

                if not _is_output_type_marked(output_type):
                    output_type = PromptedOutput(output_type)
            return super().build_agent(
                level=level,
                tier_config=tier_config,
                model=model,
                system_prompt=system_prompt,
                tools=tools,
                output_type=output_type,
                name=name,
                retries=retries,
            )

        # Tool path: use explicit model override, or resolve from tier_config
        # (falling back to baked defaults).
        if model is not None:
            sdk_model = model
        else:
            # Use the bare model name the SDK understands (e.g. "opus"), NOT the
            # transport-prefixed id (".model" == "claudeSDK-opus"), which the SDK
            # does not recognize — it yields a degenerate "error result" frame.
            # (The no-tools Model path already resolves via ``model_name``.)
            sdk_model = _resolve_model_name(tier_config, level)

        allowed_tools, server = _convert_tools(tools)
        return _SdkToolAgentHandle(
            sdk_model=sdk_model,
            system_prompt=system_prompt,
            server=server,
            allowed_tools=allowed_tools,
            output_type=output_type,
            name=name,
            workspace_root=workspace_root,
            builtin_tools=builtin_tools,
        )

"""SDK tool-agent execution path — tool conversion, workspace confinement, and
streaming-invocation logic for :class:`_SdkToolAgentHandle`.

This module is extracted from ``provider.py`` to keep that module focused on
the single public class :class:`ClaudeSDKProvider`.
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

from ..core.tracing import (
    GEN_AI_OPERATION_NAME,
    GEN_AI_PROVIDER_NAME,
    GEN_AI_REQUEST_MODEL,
    GEN_AI_SYSTEM,
    GEN_AI_USAGE_INPUT_TOKENS,
    GEN_AI_USAGE_OUTPUT_TOKENS,
    LANGFUSE_OBSERVATION_INPUT,
    LANGFUSE_OBSERVATION_METADATA_REASONING,
    LANGFUSE_OBSERVATION_OUTPUT,
    OP_CHAT,
    OP_INVOKE_AGENT,
    get_tracer,
    start_span,
)
from ._chat_messages import _chat_messages_input
from ._confinement import (
    _EDIT_TOOLS,
    _make_bash_confine_hook,
    _make_confine_hook,
)
from ._output import _build_schema_json, _parse_output
from ._task_budget import build_task_budget, run_with_task_budget

if TYPE_CHECKING:  # pragma: no cover — types-only; runtime imports stay lazy
    from claude_agent_sdk import (
        ClaudeAgentOptions,
    )

log = logging.getLogger("robotsix_llmio.claude_sdk")

_TRACER_NAME = "robotsix_llmio.claude_sdk"

# Structured-output instruction template, mirrored from pydantic-ai's
# ``prompted_output_template`` profile default.
_JSON_OUTPUT_INSTRUCTION = (
    "Always respond with a JSON object that's compatible with this schema:\n"
    "{schema}\n"
    "Don't include any text or Markdown fencing before or after."
)

# How many times to (re-)drive the SDK query for one agent run, retrying only
# transient failures (e.g. the degenerate-success frame). Small: a re-run
# clears the flaky case; a persistent error fails after the last attempt.
_SDK_QUERY_ATTEMPTS = 3

# Built-in Claude Agent SDK / Claude Code tools denied when an agent is
# restricted to ONLY its injected MCP tools (``builtin_tools=False``). A ``"*"``
# wildcard also denies the MCP tools (the SDK matches it against ``mcp__*`` too),
# so the built-ins are enumerated by name; MCP tools are namespaced ``mcp__*``
# and are therefore unaffected. Intentionally broad — listing a tool a given SDK
# version doesn't expose is harmless.
# Read-only web access. Denied with the rest of the built-ins by default, but
# separable: an agent restricted from the filesystem and shell may still have a
# legitimate need to look something up, and denying that silently is worse than
# denying it loudly. A research subsession asked to check three CVE advisories
# reported "11 sources fetched, all empty" on 2026-08-07 — the calls had been
# REFUSED, and the refusals were indistinguishable from empty results.
#
# These read nothing local and mutate nothing, so they do not reopen the
# sandbox the denylist exists to enforce.
_WEB_TOOL_NAMES = ["WebFetch", "WebSearch"]

_BUILTIN_TOOL_DENYLIST = [
    "Bash",
    "BashOutput",
    "KillShell",
    "KillBash",
    "Read",
    "Write",
    "Edit",
    "MultiEdit",
    "NotebookEdit",
    "NotebookRead",
    "Glob",
    "Grep",
    "LS",
    "Task",
    "Agent",
    "Monitor",
    "TodoWrite",
    "SlashCommand",
    "AskUserQuestion",
    "ExitPlanMode",
    "EnterPlanMode",
    "ScheduleWakeup",
    "CronCreate",
    "CronDelete",
    "CronList",
    "EnterWorktree",
    "ExitWorktree",
    "DesignSync",
    "PushNotification",
    "RemoteTrigger",
    "TaskCreate",
    "TaskGet",
    "TaskList",
    "TaskOutput",
    "TaskStop",
    "TaskUpdate",
]


@dataclass
class _SdkToolResult:
    """Minimal result mirroring pydantic-ai's ``AgentRunResult`` interface.

    .. note::

        ``all_messages()`` returns minimal history (a single final
        ``ModelResponse``) — intermediate ``ToolCallPart`` objects are not
        surfaced because the SDK owns the agent loop and tool execution.
    """

    output: Any
    _messages: list[Any]
    _usage: dict[str, Any] | None

    def all_messages(self) -> list[Any]:
        """Best-effort message history (may contain only the final text)."""
        return self._messages

    @property
    def usage(self) -> Any:
        """Aggregate token usage from the final ``ResultMessage``."""
        from ._usage import map_usage_dict

        return map_usage_dict(self._usage)


class _SdkToolAgentHandle:
    """Handle that drives the SDK tool loop directly, bypassing pydantic-ai.

    Satisfies the ``AgentHandle``-compatible contract (``run_sync``,
    ``close``).  Callers interact with the returned :class:`_SdkToolResult`.
    """

    def __init__(
        self,
        sdk_model: str,
        system_prompt: str,
        server: Any,
        allowed_tools: list[str],
        output_type: Any = str,
        name: str | None = None,
        max_turns: int | None = None,
        workspace_root: str | Path | None = None,
        builtin_tools: bool = True,
        web_tools: bool = False,
        max_tokens: int | None = None,
    ) -> None:
        self._sdk_model = sdk_model
        self._system_prompt = system_prompt
        self._server = server
        self._allowed_tools = allowed_tools
        self._output_type = output_type
        if isinstance(output_type, (list, tuple)):
            from pydantic_ai.exceptions import UserError

            raise UserError(
                "list/union output_type is not supported on the claude_sdk tool path. "
                "Wrap the types in PromptedOutput([TypeA, TypeB]) explicitly."
            )
        self._name = name or "claude_sdk agent"
        self._workspace_root = str(workspace_root) if workspace_root else None
        # When False, restrict the agent to ONLY the injected MCP tools — the
        # SDK's built-in tools (Bash/Read/Edit/Monitor/...) are denied. Use for
        # untrusted/embedded agents (e.g. a chat) that must not run shell or
        # touch the filesystem. When True (default), the agent keeps full
        # built-in tool access (coding agents), with edits confined to
        # *workspace_root* if given.
        self._builtin_tools = builtin_tools
        self._web_tools = web_tools
        self._max_tokens = max_tokens
        if max_turns is None:
            # Single source of truth for the runaway cap (see model._MAX_TURNS).
            # Generous, because an injected-MCP-tool loop legitimately needs many
            # turns; reaching it is a hard ClaudeSDKTurnLimitError, never retried.
            from ._model import _MAX_TURNS

            max_turns = _MAX_TURNS
        self._max_turns = max_turns

    def _warn_dropped_run_kwargs(
        self, model_settings: Any, kwargs: dict[str, Any]
    ) -> None:
        """Loudly flag run arguments this transport can't honor, instead of
        silently dropping them. The SDK owns the agent loop, so pydantic-ai run
        knobs (``model_settings``, ``usage_limits``, ``deps``, …) have no effect
        here — and a silently-ignored ``usage_limits`` or the like is a real
        footgun. ``message_history`` IS honored (folded into the prompt), so it
        is consumed as a named parameter and never reaches here."""
        dropped = list(kwargs)
        if model_settings is not None:
            dropped.append("model_settings")
        if dropped:
            log.warning(
                "%s: the Claude SDK tool path ignores these run arguments "
                "(the SDK runs its own agent loop): %s. Only user_prompt and "
                "message_history are honored.",
                self._name,
                ", ".join(sorted(dropped)),
            )

    def run_sync(
        self,
        user_prompt: str | list[Any],
        *,
        model_settings: dict[str, Any] | None = None,
        message_history: list[Any] | None = None,
        **kwargs: Any,
    ) -> _SdkToolResult:
        """Run the SDK tool loop synchronously, return ``_SdkToolResult``.

        *message_history* (pydantic-ai ``ModelMessage`` list) is honored: it is
        rendered into the prompt so a multi-turn caller (e.g. a review loop)
        keeps its context. Other pydantic-ai run kwargs can't be honored by the
        SDK loop and are warned about rather than silently dropped."""
        self._warn_dropped_run_kwargs(model_settings, kwargs)
        return asyncio.run(self._run(user_prompt, message_history=message_history))

    async def run(
        self,
        user_prompt: str | list[Any],
        *,
        model_settings: dict[str, Any] | None = None,
        message_history: list[Any] | None = None,
        **kwargs: Any,
    ) -> _SdkToolResult:
        """Async entry point — awaits the SDK tool loop on the current event
        loop (unlike :meth:`run_sync`, which calls ``asyncio.run``). This lets a
        tool-bearing agent be used as a subagent from inside another agent's
        tool: ``await subagent.run(...)``. Its spans then nest under the calling
        tool's span. *message_history* is honored as in :meth:`run_sync`."""
        self._warn_dropped_run_kwargs(model_settings, kwargs)
        return await self._run(user_prompt, message_history=message_history)

    def _prepare_prompt(
        self, user_prompt: Any, message_history: list[Any] | None
    ) -> tuple[str, str, list[tuple[str, bytes]]]:
        """Fold *message_history* into the prompt and augment the system prompt
        with the JSON-schema directive when ``output_type`` is structured.

        *user_prompt* may be a plain string or a pydantic-ai multimodal
        sequence (``list[str | BinaryContent]``) — image parts are split out
        and returned for :func:`~.model.build_sdk_prompt`, NEVER stringified
        (``str()`` on a ``BinaryContent`` reprs megabytes of raw bytes into
        the prompt and stalls the CLI).

        Returns ``(prompt_text, system_prompt, images)``; pure function of
        inputs + the handle's ``_system_prompt`` and ``_output_type``
        attributes.
        """
        from ._prompt import extract_prompt_parts

        system_prompt = self._system_prompt
        prompt, images = extract_prompt_parts(user_prompt)

        # Honor message_history: the SDK query is stateless per call, so fold any
        # prior pydantic-ai conversation into the prompt as a labelled transcript
        # (same rendering the no-tools ClaudeSDKModel path uses) and append the
        # new turn. Without this the tool path silently lost a caller's context.
        if message_history:
            from ._prompt import render_prompt

            history_text = render_prompt(message_history)
            if history_text:
                prompt = f"{history_text}\n\nUser: {prompt}"

        # Augment system prompt with JSON schema for structured output.
        if self._output_type is not str:
            schema_json = _build_schema_json(self._output_type)
            system_prompt = f"{system_prompt}\n\n" + _JSON_OUTPUT_INSTRUCTION.format(
                schema=schema_json
            )

        return prompt, system_prompt, images

    def _build_options(self, system_prompt: str) -> ClaudeAgentOptions:
        """Build the ``ClaudeAgentOptions`` for the SDK call, wiring in the
        workspace-confinement ``PreToolUse`` hook when a workspace root is set.
        """
        try:
            from claude_agent_sdk import (
                ClaudeAgentOptions,
            )
        except ImportError as exc:  # pragma: no cover
            raise ImportError(
                "robotsix_llmio.claude_sdk requires the 'claude_sdk' extra. "
                "Install with: pip install 'robotsix-llmio[claude_sdk]' "
                "(also needs Node.js and a logged-in `claude` CLI)."
            ) from exc

        # Confine built-in edits to the workspace when a root is set: run the
        # SDK there (so relative paths + Bash default into it) and gate every
        # Write/Edit/MultiEdit/NotebookEdit through a PreToolUse hook.
        extra: dict[str, Any] = {}
        if self._workspace_root:
            try:
                from claude_agent_sdk import HookMatcher
            except ImportError as exc:  # pragma: no cover
                raise ImportError(
                    "robotsix_llmio.claude_sdk requires the 'claude_sdk' extra. "
                    "Install with: pip install 'robotsix-llmio[claude_sdk]' "
                    "(also needs Node.js and a logged-in `claude` CLI)."
                ) from exc

            extra["cwd"] = self._workspace_root
            extra["hooks"] = {
                "PreToolUse": [
                    HookMatcher(
                        matcher=_EDIT_TOOLS,
                        hooks=[_make_confine_hook(self._workspace_root)],
                    ),
                    HookMatcher(
                        matcher="Bash",
                        hooks=[_make_bash_confine_hook(self._workspace_root)],
                    ),
                ]
            }

        # ``bypassPermissions`` auto-approves tool use (no headless approval
        # stall, which otherwise degenerates into a spurious "error result").
        extra["permission_mode"] = "bypassPermissions"
        if self._builtin_tools:
            extra["allowed_tools"] = self._allowed_tools
        else:
            # Restricted: expose ONLY the injected MCP tools. ``disallowed_tools``
            # is the reliable lever (``allowed_tools`` / ``can_use_tool`` are not
            # enforced for built-ins). We deny the built-ins by explicit name
            # rather than ``"*"`` — the wildcard would also deny the ``mcp__*``
            # tools — leaving the injected MCP tools (registered via
            # ``mcp_servers``) callable. ``allowed_tools`` is intentionally unset.
            denied = list(_BUILTIN_TOOL_DENYLIST)
            if not self._web_tools:
                denied += _WEB_TOOL_NAMES
            extra["disallowed_tools"] = denied

        return ClaudeAgentOptions(
            system_prompt=system_prompt,
            model=self._sdk_model,
            max_turns=self._max_turns,
            mcp_servers={"milltools": self._server},
            setting_sources=[],
            task_budget=build_task_budget(
                self._max_tokens, self._name, self._sdk_model
            ),
            **extra,
        )

    async def _invoke_query(
        self, prompt: str | list[dict[str, Any]], options: ClaudeAgentOptions
    ) -> tuple[str, Any, str]:
        """Run the SDK streaming loop under the per-call wall-clock cap.

        Returns ``(text, result, reasoning)`` — joined assistant text (with the
        ``ResultMessage.result`` fallback), the captured ``ResultMessage``, and
        the model's joined extended-thinking content (``""`` when off).
        Converts a wall-clock timeout into :class:`ClaudeSDKQueryTimeout`.

        Transient SDK failures — notably the degenerate ``is_error=True`` /
        ``subtype="success"`` frame, which a re-run clears — are retried here.
        The no-tools Model path gets this for free via pydantic-ai's retries;
        the tool path drives the SDK loop itself, so it must retry on its own.

        The whole retry loop is wrapped in :func:`run_with_task_budget` rather
        than each attempt: a model that rejects ``task_budget`` rejects it on
        every attempt, so discovering that inside the loop would burn the
        transient budget re-sending the same invalid request. Wrapping outside
        means the budget is dropped once and the loop then runs with its full
        allowance against a request that can actually succeed.
        """
        from ._stream import _stream_query
        from .transient import is_claude_sdk_transient

        async def _run(opts: ClaudeAgentOptions) -> tuple[str, Any, str]:
            last_exc: Exception | None = None
            for attempt in range(_SDK_QUERY_ATTEMPTS):
                try:
                    return await _stream_query(prompt, opts, self._name)
                except Exception as exc:
                    last_exc = exc
                    if attempt + 1 < _SDK_QUERY_ATTEMPTS and is_claude_sdk_transient(
                        exc
                    ):
                        log.warning(
                            "%s: transient SDK error on attempt %d/%d, retrying: %s",
                            self._name,
                            attempt + 1,
                            _SDK_QUERY_ATTEMPTS,
                            exc,
                        )
                        continue
                    raise
            assert last_exc is not None  # unreachable: loop returns or raises
            raise last_exc

        return await run_with_task_budget(_run, options, self._sdk_model, self._name)

    def _record_generation_span(
        self,
        system_prompt: str,
        prompt: str,
        text: str,
        result: Any,
        reasoning: str = "",
    ) -> None:
        """Open the child ``chat {model}`` generation span and stamp token
        usage + the SDK cost estimate on it.

        Cost MUST sit on this child observation to roll up — a root span
        becomes the trace, not a summable observation. *reasoning* (the model's
        extended-thinking content) is recorded as observation metadata when
        present, so traces show the model's reasoning, not just the answer and
        tool calls.
        """
        from ..core.cost import record_cost
        from ._model import PROVIDER_NAME
        from ._usage import _best_usage_dict

        # Child generation span: the model exchange. Carries input/output +
        # token usage + the SDK cost estimate. Cost must sit on a child
        # observation to roll up — a root span becomes the trace, not summed.
        usage_obj = _best_usage_dict(result)
        with start_span(
            get_tracer(_TRACER_NAME),
            f"chat {self._sdk_model}",
            {
                GEN_AI_OPERATION_NAME: OP_CHAT,
                GEN_AI_PROVIDER_NAME: PROVIDER_NAME,
                GEN_AI_SYSTEM: "anthropic",
                GEN_AI_REQUEST_MODEL: self._sdk_model,
                # Record system + user as chat messages so the system prompt
                # (sent to the SDK, but previously absent from traces) shows
                # up on the generation in Langfuse.
                LANGFUSE_OBSERVATION_INPUT: _chat_messages_input(system_prompt, prompt),
                LANGFUSE_OBSERVATION_OUTPUT: text,
            },
        ) as gen:
            if gen is not None:
                if reasoning:
                    gen.set_attribute(
                        LANGFUSE_OBSERVATION_METADATA_REASONING, reasoning
                    )
                if isinstance(usage_obj, dict):
                    in_tok = usage_obj.get("input_tokens")
                    out_tok = usage_obj.get("output_tokens")
                    if in_tok is not None:
                        gen.set_attribute(GEN_AI_USAGE_INPUT_TOKENS, int(in_tok))
                    if out_tok is not None:
                        gen.set_attribute(GEN_AI_USAGE_OUTPUT_TOKENS, int(out_tok))
            record_cost(
                result,
                lambda r: getattr(r, "total_cost_usd", None),
                provider=PROVIDER_NAME,
            )

    async def _run(
        self, user_prompt: str | list[Any], message_history: list[Any] | None = None
    ) -> _SdkToolResult:
        from ._model import PROVIDER_NAME
        from ._usage import _best_usage_dict

        prompt, system_prompt, images = self._prepare_prompt(
            user_prompt, message_history
        )
        options = self._build_options(system_prompt)
        if images:
            log.info(
                "%s: attaching %d image block(s) via streaming input",
                self._name,
                len(images),
            )
        root_attrs = {
            GEN_AI_OPERATION_NAME: OP_INVOKE_AGENT,
            GEN_AI_PROVIDER_NAME: PROVIDER_NAME,
            GEN_AI_SYSTEM: "anthropic",
            GEN_AI_REQUEST_MODEL: self._sdk_model,
            # This span becomes the trace, so render system + user as chat messages
            # here too — system prompt then shows at the trace root, not just child.
            LANGFUSE_OBSERVATION_INPUT: _chat_messages_input(system_prompt, prompt),
        }
        with start_span(get_tracer(_TRACER_NAME), self._name, root_attrs) as root:
            log.info(
                "%s: starting (model=%s, max_turns=%d)",
                self._name,
                self._sdk_model,
                self._max_turns,
            )
            from ._prompt import build_sdk_prompt

            text, result, reasoning = await self._invoke_query(
                build_sdk_prompt(prompt, images), options
            )
            if root is not None:
                root.set_attribute(LANGFUSE_OBSERVATION_OUTPUT, text)
            self._record_generation_span(system_prompt, prompt, text, result, reasoning)
        from pydantic_ai.messages import ModelResponse, TextPart

        return _SdkToolResult(
            output=_parse_output(text, self._output_type),
            _messages=[ModelResponse(parts=[TextPart(content=text)])],
            _usage=_best_usage_dict(result),
        )

    def close(self) -> None:
        """No-op — no HTTP client to close."""
        pass

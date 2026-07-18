"""Tool conversion — pydantic-ai tools → SDK MCP tool format.

Pure transformation with no references to agent state.
"""

from __future__ import annotations

import inspect
import json
from typing import TYPE_CHECKING, Any

from pydantic_ai.exceptions import UserError

from ..core.tracing import (
    GEN_AI_OPERATION_NAME,
    GEN_AI_TOOL_NAME,
    LANGFUSE_OBSERVATION_INPUT,
    LANGFUSE_OBSERVATION_OUTPUT,
    OP_EXECUTE_TOOL,
    get_tracer,
    start_span,
)

if TYPE_CHECKING:  # pragma: no cover — types-only; runtime imports stay lazy
    pass

_TRACER_NAME = "robotsix_llmio.claude_sdk"


def _convert_tools(tools: list[Any]) -> tuple[list[str], Any]:
    """Convert pydantic-ai tools into SDK MCP tools.

    Returns:
        ``(allowed_tools, mcp_server)`` — the *allowed_tools* entries
        (``"mcp__milltools__<name>"``) and the MCP server object to pass
        to ``ClaudeAgentOptions.mcp_servers``.

    """
    import pydantic_ai

    try:
        from claude_agent_sdk import (
            create_sdk_mcp_server,
        )
        from claude_agent_sdk import tool as sdk_tool
    except ImportError as exc:
        raise ImportError(
            "robotsix_llmio.claude_sdk requires the 'claude_sdk' extra. "
            "Install with: pip install 'robotsix-llmio[claude_sdk]' "
            "(also needs Node.js and a logged-in `claude` CLI)."
        ) from exc

    wrapped: list[Any] = []
    allowed: list[str] = []

    for t in tools:
        # Normalize: plain callables become pydantic_ai.Tool (idempotent).
        if not isinstance(t, pydantic_ai.Tool):
            t = pydantic_ai.Tool(t)

        if t.takes_ctx:
            sig = inspect.signature(t.function_schema.function)
            ctx_param = sig.parameters.get("ctx")
            if (
                ctx_param is not None
                and ctx_param.default is not inspect.Parameter.empty
            ):
                # Optional ctx — the function will be called without ctx,
                # letting the default value apply (e.g. ctx=None).
                pass
            else:
                raise UserError(
                    f"ClaudeSDKModel does not support tools that take a required "
                    f"RunContext (tool {t.name!r} has takes_ctx=True with no default "
                    f"for ctx): the Claude Agent SDK invokes tools with only their "
                    f"JSON arguments, so no run context can be supplied. Rewrite the "
                    f"tool to accept an optional ctx parameter (e.g. "
                    f"ctx: RunContext[None] = None) or to take plain arguments only."
                )

        name: str = t.name
        # The SDK's @tool wants a str description; pydantic-ai's may be None.
        description: str = t.description or ""
        schema: dict[str, Any] = t.tool_def.parameters_json_schema
        fn = t.function_schema.function
        is_async: bool = t.function_schema.is_async

        @sdk_tool(name, description, schema)  # type: ignore[untyped-decorator, unused-ignore]
        async def _wrapper(
            args: dict[str, Any],
            _fn: Any = fn,
            _is_async: bool = is_async,
            _name: str = name,
        ) -> dict[str, Any]:
            # Emit a TOOL span around the actual call, so the tool (and any
            # subagent it runs) nests under the agent-run span in traces.
            with start_span(
                get_tracer(_TRACER_NAME),
                _name,
                {
                    GEN_AI_OPERATION_NAME: OP_EXECUTE_TOOL,
                    GEN_AI_TOOL_NAME: _name,
                    LANGFUSE_OBSERVATION_INPUT: json.dumps(args, default=str),
                },
            ) as sp:
                if _is_async:
                    result = await _fn(**args)
                else:
                    result = _fn(**args)
                if sp is not None:
                    sp.set_attribute(LANGFUSE_OBSERVATION_OUTPUT, str(result))
                return {"content": [{"type": "text", "text": str(result)}]}

        wrapped.append(_wrapper)
        allowed.append(f"mcp__milltools__{name}")

    server = create_sdk_mcp_server(name="milltools", tools=wrapped)
    return allowed, server

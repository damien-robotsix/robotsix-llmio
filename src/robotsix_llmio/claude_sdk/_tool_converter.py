"""Tool conversion — pydantic-ai tools → SDK MCP tool format.

Pure transformation with no references to agent state.
"""

from __future__ import annotations

import base64
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
from ._prompt import _binary_placeholder, _is_binary_part, _is_image_part

if TYPE_CHECKING:  # pragma: no cover — types-only; runtime imports stay lazy
    pass

_TRACER_NAME = "robotsix_llmio.claude_sdk"


def _text_of(item: Any) -> str | None:
    """The plain text carried by a content part, or ``None``.

    Covers pydantic-ai's two spellings — ``TextPart.content`` and
    ``TextContent.content`` — plus anything exposing a ``.text`` string.
    """
    for attr in ("text", "content"):
        value = getattr(item, attr, None)
        if isinstance(value, str):
            return value
    return None


def _flatten_result(result: Any) -> list[Any]:
    """Flatten a tool return value into a list of content items.

    A pydantic-ai ``ToolReturn`` is unwrapped into its ``return_value``
    followed by its extra ``content``; both can carry binary parts, and
    reprs of either would leak raw bytes into the prompt.
    """
    items: list[Any] = []
    for item in result if isinstance(result, (list, tuple)) else [result]:
        if getattr(item, "kind", None) == "tool-return" and hasattr(
            item, "return_value"
        ):
            items.extend(_flatten_result(item.return_value))
            extra = getattr(item, "content", None)
            if extra is not None:
                items.extend(_flatten_result(extra))
        else:
            items.append(item)
    return items


def _mcp_content_blocks(result: Any) -> list[dict[str, Any]]:
    """Map a tool's return value onto MCP content blocks.

    Image payloads (pydantic-ai ``BinaryContent``, or anything exposing
    ``data`` bytes with an ``image/*`` ``media_type``) become native MCP
    ``image`` blocks, so a vision-capable model actually SEES them.  The
    SDK's in-process MCP server converts those straight to
    ``ImageContent``.

    Raw bytes must NEVER be stringified: ``str()`` on a ``BinaryContent``
    reprs the payload, so a 200 KB PNG lands in the prompt as a ~590 KB
    blob of escaped byte escapes — unreadable to the model and big enough
    to swamp the context window.  That is what a tool returning a
    rendered page used to produce.

    Returns for tools with no binary payload are left exactly as they
    were: a single ``text`` block holding ``str(result)``.
    """
    items = _flatten_result(result)
    if not any(_is_binary_part(i) for i in items):
        return [{"type": "text", "text": str(result)}]

    blocks: list[dict[str, Any]] = []
    for item in items:
        if _is_image_part(item):
            blocks.append(
                {
                    "type": "image",
                    "data": base64.b64encode(bytes(item.data)).decode("ascii"),
                    "mimeType": item.media_type,
                }
            )
        elif _is_binary_part(item):
            # Audio/document payloads have no MCP block the CLI accepts —
            # describe them instead of reprising the bytes.
            blocks.append({"type": "text", "text": _binary_placeholder(item)})
        else:
            text = _text_of(item)
            blocks.append(
                {"type": "text", "text": text if text is not None else str(item)}
            )
    return blocks


def _span_output(blocks: list[dict[str, Any]]) -> str:
    """Trace-friendly rendering of MCP content *blocks*.

    Image blocks collapse to a size summary so a base64 payload never
    reaches the span attribute (and from there Langfuse).
    """
    parts: list[str] = []
    for block in blocks:
        if block.get("type") == "image":
            parts.append(
                f"[image: {block.get('mimeType', 'unknown')}, "
                f"{len(block.get('data', ''))} base64 chars]"
            )
        else:
            parts.append(block.get("text", ""))
    return "\n".join(parts)


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
                blocks = _mcp_content_blocks(result)
                if sp is not None:
                    sp.set_attribute(LANGFUSE_OBSERVATION_OUTPUT, _span_output(blocks))
                return {"content": blocks}

        wrapped.append(_wrapper)
        allowed.append(f"mcp__milltools__{name}")

    server = create_sdk_mcp_server(name="milltools", tools=wrapped)
    return allowed, server

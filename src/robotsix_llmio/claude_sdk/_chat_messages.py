"""Chat-messages rendering for tracing — pure data transformation."""

from __future__ import annotations

import json


def _chat_messages_input(system_prompt: str, user_text: str) -> str:
    """JSON ``{role, content}`` message list (system + user) for a generation
    span's Langfuse input.

    The system prompt IS sent to the SDK (``ClaudeAgentOptions.system_prompt``),
    but the span previously recorded only the user prompt — so traces showed the
    input without the system. Rendering both as chat messages surfaces the
    system prompt in Langfuse (which parses the JSON and shows the roles), the
    same shape the OpenRouter/pydantic-ai path produces."""
    return json.dumps(
        [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_text},
        ],
        default=str,
    )

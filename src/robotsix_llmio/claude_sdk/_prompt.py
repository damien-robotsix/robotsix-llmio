"""Prompt helpers for the Claude Agent SDK transport.

Pure functions with no ``ClaudeSDKModel`` dependency — text extraction, binary
placeholder generation, image collection, prompt rendering, and usage mapping.
"""

from __future__ import annotations

from typing import Any, cast

from pydantic_ai.messages import (
    ModelMessage,
    ModelRequest,
    ModelResponse,
    RetryPromptPart,
    TextPart,
    ToolReturnPart,
    UserPromptPart,
)
from pydantic_ai.usage import RequestUsage

from ._usage import _best_usage_dict, map_usage_dict


def _binary_placeholder(item: Any) -> str:
    """Compact stand-in for a binary content part (image/audio/document).

    The SDK path is text-only, so binary payloads cannot reach the model —
    but they must NEVER be stringified either: ``str()`` on a pydantic-ai
    ``BinaryContent`` reprs the raw bytes (a 2 MB image becomes a ~6 MB
    escaped-byte string), which stalls the CLI subprocess for the full
    per-call wall-clock cap and hangs the caller.
    """
    media_type = getattr(item, "media_type", None) or "unknown type"
    size = len(getattr(item, "data", b"") or b"")
    return (
        f"[binary attachment: {media_type}, {size} bytes — "
        f"not visible to this text-only model]"
    )


def _is_binary_part(item: Any) -> bool:
    """True for content parts carrying a raw-bytes payload (``BinaryContent``)."""
    return isinstance(getattr(item, "data", None), (bytes, bytearray))


def _is_image_part(item: Any) -> bool:
    """True for binary parts whose media type is an image the SDK can display."""
    media_type = getattr(item, "media_type", "") or ""
    return _is_binary_part(item) and media_type.startswith("image/")


def extract_prompt_parts(user_prompt: Any) -> tuple[str, list[tuple[str, bytes]]]:
    """Split a caller prompt into ``(text, images)``.

    *user_prompt* may be a plain string, a bare content part, or a sequence of
    parts (pydantic-ai style: ``list[str | BinaryContent]``).  Image parts are
    returned as ``(media_type, raw_bytes)`` pairs for
    :func:`build_sdk_prompt`; non-image binary parts degrade to the
    :func:`_binary_placeholder` text.
    """
    if isinstance(user_prompt, str):
        return user_prompt, []
    if _is_image_part(user_prompt):
        return "", [(user_prompt.media_type, bytes(user_prompt.data))]
    if _is_binary_part(user_prompt):
        return _binary_placeholder(user_prompt), []
    if not isinstance(user_prompt, (list, tuple)):
        return str(user_prompt), []
    texts: list[str] = []
    images: list[tuple[str, bytes]] = []
    for item in user_prompt:
        if isinstance(item, str):
            texts.append(item)
        elif _is_image_part(item):
            images.append((item.media_type, bytes(item.data)))
        elif _is_binary_part(item):
            texts.append(_binary_placeholder(item))
        else:
            text = getattr(item, "text", None)
            texts.append(text if isinstance(text, str) else str(item))
    return "\n".join(texts), images


def build_sdk_prompt(
    text: str, images: list[tuple[str, bytes]]
) -> str | list[dict[str, Any]]:
    """Build the ``query()`` prompt: plain text, or streaming-input messages.

    With no images the text passes through unchanged (the SDK's simple string
    mode).  With images, returns a list of streaming-input message dicts — one
    user message whose content carries the text plus one base64 ``image``
    block per attachment, which the CLI accepts exactly like a pasted image.
    A **list** (not an async generator) so retries can safely re-send it;
    the stream layer wraps it in a fresh async iterator per attempt.
    """
    if not images:
        return text
    import base64

    content: list[dict[str, Any]] = []
    if text:
        content.append({"type": "text", "text": text})
    for media_type, data in images:
        content.append(
            {
                "type": "image",
                "source": {
                    "type": "base64",
                    "media_type": media_type,
                    "data": base64.b64encode(data).decode("ascii"),
                },
            }
        )
    return [{"type": "user", "message": {"role": "user", "content": content}}]


def collect_latest_user_images(
    messages: list[ModelMessage],
) -> list[tuple[str, bytes]]:
    """Image parts of the newest user turn, as ``(media_type, bytes)`` pairs.

    Only the latest ``UserPromptPart`` matters: earlier turns are replayed as
    text transcript (their images already degraded to placeholders when they
    were current).
    """
    for message in reversed(messages):
        if not isinstance(message, ModelRequest):
            continue
        for part in reversed(message.parts):
            if isinstance(part, UserPromptPart):
                content = part.content
                if isinstance(content, str):
                    return []
                _text, images = extract_prompt_parts(list(content))
                return images
    return []


def _content_to_text(content: Any) -> str:
    """Flatten a pydantic-ai user/tool content (str or a list of parts) to text.

    Binary parts are replaced with a short placeholder — see
    :func:`_binary_placeholder` for why they must not be stringified.
    """
    if isinstance(content, str):
        return content
    if _is_binary_part(content):
        return _binary_placeholder(content)
    if isinstance(content, (list, tuple)):
        out: list[str] = []
        for item in content:  # heterogeneous content parts
            text = getattr(item, "text", None)
            if isinstance(text, str):
                out.append(text)
            elif _is_binary_part(item):
                out.append(_binary_placeholder(item))
            else:
                out.append(str(item))
        return "\n".join(out)
    return str(content)


def _retry_text(part: RetryPromptPart) -> str:
    """The corrective text pydantic-ai wants shown back to the model on a retry
    (e.g. a JSON-validation failure during PromptedOutput)."""
    model_response = getattr(part, "model_response", None)
    if callable(model_response):
        try:
            return cast(str, model_response())
        except Exception:  # pragma: no cover - defensive  # noqa: S110
            pass
    return _content_to_text(getattr(part, "content", ""))


def render_prompt(messages: list[ModelMessage]) -> str:
    """Flatten the pydantic-ai message history into a single prompt string for
    the (stateless-per-call) SDK ``query``. A lone user turn is sent verbatim;
    multi-turn history is rendered as a labelled transcript so the model sees
    its own prior attempt and any correction."""
    turns: list[tuple[str, str]] = []
    for message in messages:
        if isinstance(message, ModelRequest):
            for part in message.parts:
                if isinstance(part, UserPromptPart):
                    turns.append(("user", _content_to_text(part.content)))
                elif isinstance(part, ToolReturnPart):
                    turns.append(
                        (
                            "user",
                            f"Tool result ({part.tool_name}): "
                            f"{_content_to_text(part.content)}",
                        )
                    )
                elif isinstance(part, RetryPromptPart):
                    turns.append(("user", _retry_text(part)))
        elif isinstance(message, ModelResponse):
            text = "\n".join(
                p.content for p in message.parts if isinstance(p, TextPart)
            )
            if text:
                turns.append(("assistant", text))

    if len(turns) == 1 and turns[0][0] == "user":
        return turns[0][1]
    return "\n\n".join(
        f"{'User' if role == 'user' else 'Assistant'}: {text}" for role, text in turns
    )


def _map_usage(result: Any) -> RequestUsage:
    """Map a Claude Agent SDK ``ResultMessage.usage`` (or ``model_usage``)
    dict onto pydantic-ai's :class:`RequestUsage`.

    Defensive: a missing/partial dict yields zeros.
    """
    usage = _best_usage_dict(result)
    return map_usage_dict(usage)

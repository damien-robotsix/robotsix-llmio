"""Structured output parsing — JSON extraction, schema building, type validation.

Pure data-transformation functions with no references to agent state or the SDK.
"""

from __future__ import annotations

import json
import re
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:  # pragma: no cover — types-only; runtime imports stay lazy
    pass


def _output_validators(output_type: Any) -> list[Any]:
    """Return the ordered list of pydantic model classes for *output_type*."""
    from pydantic_ai import PromptedOutput

    if isinstance(output_type, PromptedOutput):
        outputs = output_type.outputs
        if isinstance(outputs, (list, tuple)):
            return list(outputs)
        return [outputs]
    return [output_type]


def _build_schema_json(output_type: Any) -> str:
    """Build the JSON schema string to embed in the system prompt.

    Single model → its own schema.  Multiple models → ``{"anyOf": [...]}``,
    matching the shape pydantic-ai uses for union output in prompted mode.
    """
    validators = _output_validators(output_type)
    if len(validators) == 1:
        return json.dumps(validators[0].model_json_schema())
    return json.dumps({"anyOf": [v.model_json_schema() for v in validators]})


def _fenced_blocks(text: str) -> list[str]:
    """Return the inner contents of every ```-fenced code block in *text*.

    Matches ``` optionally tagged with a language (```json) and captures the
    body. Models frequently wrap a structured answer in a ```json fence after
    a prose preamble, so the fence is the most reliable delimiter.
    """
    return re.findall(r"```(?:[a-zA-Z0-9_-]+)?\s*\n?(.*?)```", text, re.DOTALL)


def _balanced_objects(text: str) -> list[str]:
    """Return every top-level balanced ``{...}`` substring of *text*.

    A hand-rolled brace matcher that tracks JSON string state (so braces and
    quotes inside string literals don't throw off the depth count). Unbalanced
    stray braces in prose (``the {x} kwarg``) yield a short candidate that
    simply fails to JSON-parse later; the real object is captured whole,
    including nested objects/arrays. This replaces a naive
    ``re.search(r"\\{.*\\}")`` that anchored on the FIRST prose brace and so
    swallowed prose + JSON into one unparsable blob.
    """
    out: list[str] = []
    i, n = 0, len(text)
    while i < n:
        if text[i] != "{":
            i += 1
            continue
        depth = 0
        in_str = False
        esc = False
        j = i
        while j < n:
            c = text[j]
            if in_str:
                if esc:
                    esc = False
                elif c == "\\":
                    esc = True
                elif c == '"':
                    in_str = False
            elif c == '"':
                in_str = True
            elif c == "{":
                depth += 1
            elif c == "}":
                depth -= 1
                if depth == 0:
                    out.append(text[i : j + 1])
                    break
            j += 1
        i = j + 1
    return out


def _extract_json_object(text: str) -> dict[str, Any] | None:
    """Best-effort extraction of a JSON object from model *text*.

    Tries, in order: (1) the whole text as JSON; (2) each ```-fenced block,
    last first (a prose preamble + trailing ```json fence is the common shape);
    (3) each top-level balanced ``{...}`` substring, last first. Returns the
    first candidate that parses to a ``dict``, else ``None``.
    """
    candidates: list[str] = [text]
    candidates += list(reversed(_fenced_blocks(text)))
    candidates += list(reversed(_balanced_objects(text)))
    for cand in candidates:
        cand = cand.strip()
        if not cand:
            continue
        try:
            data = json.loads(cand)
        except json.JSONDecodeError, ValueError:
            continue
        if isinstance(data, dict):
            return data
    return None


def _parse_output(text: str, output_type: Any) -> Any:
    """Parse final assistant text against *output_type*.

    ``str`` → text as-is.  Otherwise extract a JSON object (tolerating a prose
    preamble and/or a ```json fence) and validate against each declared output
    validator in order, returning the first one that validates successfully.
    Raises ``ValueError`` when no JSON object can be found (instead of silently
    returning raw text to a structured-type caller).
    """
    if output_type is str:
        return text

    validators = _output_validators(output_type)
    data = _extract_json_object(text)
    if not isinstance(data, dict):
        raise ValueError(
            f"_parse_output: no JSON object found in model response; "
            f"expected a JSON object matching the declared output type. "
            f"Response text: {text!r}"
        )
    if not validators:
        return data

    last_exc: Exception | None = None
    for v in validators:
        try:
            return v.model_validate(data)
        except Exception as exc:
            last_exc = exc
    raise last_exc  # type: ignore[misc]

"""Provider-model identifier parser — leaf module with no intra-package imports.

Parses a combined ``provider-model`` tier identifier (e.g.
``claudeSDK-opus`` or ``openrouter[deepseek]-deepseek/deepseek-v4-flash``)
into its provider prefix, optional sub-alias, and concrete model name.

Isolated as a leaf (only imports :class:`RobotsixLLMIOError` from the
top-level ``exceptions`` module) so that both ``config.tier`` and
``core.factory`` can import it without creating import cycles.
"""

from __future__ import annotations

from typing import NamedTuple

from robotsix_llmio.exceptions import RobotsixLLMIOError


class MalformedIdentifierError(RobotsixLLMIOError):
    """Raised when a provider-model identifier string is malformed."""


class ParsedIdentifier(NamedTuple):
    """Result of :func:`parse_model_identifier`.

    Attributes
    ----------
    provider:
        The hyphen-free provider prefix (e.g. ``"claudeSDK"`` or
        ``"openrouter"``).
    sub_alias:
        The optional bracketed sub-alias (e.g. ``"deepseek"`` for
        ``"openrouter[deepseek]"``), or ``None``.
    model_name:
        Everything after the first out-of-bracket hyphen —
        the concrete model name fed to the backend.  May contain
        hyphens and slashes (e.g. ``"deepseek/deepseek-v4-flash"``).
    """

    provider: str
    sub_alias: str | None
    model_name: str


def parse_model_identifier(identifier: str) -> ParsedIdentifier:
    """Parse a combined provider-model tier identifier.

    Grammar::

        <provider-prefix>-<model-name>

    where ``<provider-prefix>`` is ``<provider>`` or
    ``<provider>[<sub-alias>]``, ``<provider>`` contains no hyphen, and
    ``<model-name>`` is everything after the **first hyphen that is
    outside any bracket** (it may itself contain hyphens and slashes).

    Parameters
    ----------
    identifier:
        The combined identifier string — e.g. ``"claudeSDK-opus"`` or
        ``"openrouter[deepseek]-deepseek/deepseek-v4-flash"``.

    Returns
    -------
    ParsedIdentifier
        Parsed components.

    Raises
    ------
    MalformedIdentifierError
        If the identifier has no hyphen-delimited model part, an empty
        provider or model, or unbalanced / empty brackets.
    """
    # ---------- find the first hyphen outside any bracket ----------
    bracket_depth = 0
    hyphen_idx = -1
    for i, ch in enumerate(identifier):
        if ch == "[":
            bracket_depth += 1
        elif ch == "]":
            bracket_depth -= 1
            if bracket_depth < 0:
                raise MalformedIdentifierError(
                    f"Unbalanced brackets in identifier {identifier!r}"
                )
        elif ch == "-" and bracket_depth == 0:
            hyphen_idx = i
            break

    if hyphen_idx == -1:
        # Check for unbalanced brackets before reporting no hyphen.
        if bracket_depth != 0:
            raise MalformedIdentifierError(
                f"Unbalanced brackets in identifier {identifier!r}"
            )
        raise MalformedIdentifierError(
            f"No hyphen-delimited model part in identifier {identifier!r}"
        )

    prefix = identifier[:hyphen_idx]
    model_name = identifier[hyphen_idx + 1 :]

    if not prefix:
        raise MalformedIdentifierError(
            f"Empty provider prefix in identifier {identifier!r}"
        )
    if not model_name:
        raise MalformedIdentifierError(f"Empty model name in identifier {identifier!r}")

    # Remaining brackets must be balanced (already checked depth≥0 on close,
    # now confirm depth is exactly zero at end).
    if bracket_depth != 0:
        raise MalformedIdentifierError(
            f"Unbalanced brackets in identifier {identifier!r}"
        )

    # ---------- extract provider and optional sub-alias ----------
    provider: str
    sub_alias: str | None = None

    bracket_start = prefix.find("[")
    if bracket_start != -1:
        bracket_end = prefix.find("]", bracket_start)
        if bracket_end == -1:  # pragma: no cover — caught by depth check above
            raise MalformedIdentifierError(
                f"Unbalanced brackets in identifier {identifier!r}"
            )
        provider = prefix[:bracket_start]
        sub_alias = prefix[bracket_start + 1 : bracket_end]
        if not provider:
            raise MalformedIdentifierError(
                f"Empty provider in identifier {identifier!r}"
            )
        if not sub_alias:
            raise MalformedIdentifierError(
                f"Empty sub-alias in brackets in identifier {identifier!r}"
            )
        # Nothing may follow the closing bracket.
        if bracket_end + 1 < len(prefix):
            raise MalformedIdentifierError(
                f"Unexpected content after sub-alias brackets in"
                f" identifier {identifier!r}"
            )
    else:
        provider = prefix

    return ParsedIdentifier(
        provider=provider,
        sub_alias=sub_alias,
        model_name=model_name,
    )

"""Provider-model identifier parser — leaf module with no intra-package imports.

Parses a combined ``provider-model`` tier identifier (e.g.
``claudeSDK-opus`` or ``openrouter-deepseek/deepseek-v4-flash-latest``) into its
provider prefix and concrete model name, split on the first hyphen.

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

    Attributes:
        provider: The provider prefix before the first hyphen (e.g.
            ``"claudeSDK"`` or ``"openrouter"``).
        model_name: Everything after the first hyphen — the concrete model
            name fed to the backend.  May contain hyphens and slashes
            (e.g. ``"deepseek/deepseek-v4-flash-latest"``).

    """

    provider: str
    model_name: str


def parse_model_identifier(identifier: str) -> ParsedIdentifier:
    """Parse a combined provider-model tier identifier.

    Grammar::

        <provider>-<model-name>

    where ``<provider>`` is everything before the **first hyphen** and
    ``<model-name>`` is everything after it (the model name may itself
    contain hyphens and slashes).

    Args:
        identifier: The combined identifier string — e.g.
            ``"claudeSDK-opus"`` or
            ``"openrouter-deepseek/deepseek-v4-flash-latest"``.

    Returns:
        ParsedIdentifier: Parsed components.

    Raises:
        MalformedIdentifierError: If the identifier has no
            hyphen-delimited model part, or an empty provider or model
            name.

    """
    provider, sep, model_name = identifier.partition("-")
    if not sep:
        raise MalformedIdentifierError(
            f"No hyphen-delimited model part in identifier {identifier!r}"
        )
    if not provider:
        raise MalformedIdentifierError(
            f"Empty provider prefix in identifier {identifier!r}"
        )
    if not model_name:
        raise MalformedIdentifierError(f"Empty model name in identifier {identifier!r}")

    return ParsedIdentifier(provider=provider, model_name=model_name)

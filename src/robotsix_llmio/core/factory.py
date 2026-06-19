"""Provider-agnostic factory: select a backend by provider-prefix.

Consumers obtain a provider through :func:`get_provider_for_identifier` and
never import a concrete provider class.  The provider prefix is parsed from
a combined ``provider-model`` tier identifier (e.g. ``"claudeSDK-opus"``)
and the backend is lazy-imported via :data:`_PROVIDER_PREFIX_MAP`.

Provider classes are **lazy-imported** on resolution so optional extras stay
optional: a missing extra surfaces an :class:`ImportError` naming the extra and
the exact ``pip install`` command.
"""

from __future__ import annotations

import importlib
from dataclasses import dataclass
from typing import Any

from .provider import LLMProvider


@dataclass(frozen=True)
class _ProviderEntry:
    """Lazy-import coordinates of a registered provider class."""

    module: str
    class_name: str
    extra: str


# ---------------------------------------------------------------------------
#  Provider-prefix → backend map (single source of truth)
# ---------------------------------------------------------------------------

_PROVIDER_PREFIX_MAP: dict[str, _ProviderEntry] = {
    "claudeSDK": _ProviderEntry(
        module="robotsix_llmio.claude_sdk",
        class_name="ClaudeSDKProvider",
        extra="claude_sdk",
    ),
    "openrouter": _ProviderEntry(
        module="robotsix_llmio.openrouter_deepseek",
        class_name="OpenRouterDeepseekProvider",
        extra="openrouter_deepseek",
    ),
}
"""Provider-prefix → lazy-import coordinates.

Keys are the hyphen-free provider prefixes parsed from a combined
``provider-model`` tier identifier (e.g. ``"claudeSDK"`` or
``"openrouter"``).  This is the **single source of truth** that maps a
prefix to the backend that serves it.
"""


def get_provider_for_identifier(identifier: str, **kwargs: Any) -> LLMProvider:
    """Resolve and instantiate a provider from a combined tier identifier.

    The *identifier* is parsed via :func:`~.identifier.parse_model_identifier`,
    the provider prefix looked up in :data:`_PROVIDER_PREFIX_MAP`, and the
    backend is lazy-imported.  ``**kwargs`` are forwarded to the provider
    constructor.

    Parameters
    ----------
    identifier:
        Combined provider-model identifier — e.g.
        ``"claudeSDK-opus"`` or
        ``"openrouter[deepseek]-deepseek/deepseek-v4-flash"``.
    **kwargs:
        Forwarded to the resolved provider class constructor.

    Returns
    -------
    LLMProvider
        A fully-instantiated provider.

    Raises
    ------
    MalformedIdentifierError
        If *identifier* cannot be parsed.
    ValueError
        If the parsed provider prefix is not in :data:`_PROVIDER_PREFIX_MAP`.
    ImportError
        If the provider's optional extra is not installed.
    """
    from .identifier import parse_model_identifier

    parsed = parse_model_identifier(identifier)
    prefix = parsed.provider

    try:
        entry = _PROVIDER_PREFIX_MAP[prefix]
    except KeyError as exc:
        known = ", ".join(sorted(_PROVIDER_PREFIX_MAP))
        raise ValueError(
            f"Unknown provider prefix {prefix!r}. Known prefixes: {known}."
        ) from exc

    try:
        mod = importlib.import_module(entry.module)
        cls = getattr(mod, entry.class_name)
    except ImportError as exc:
        raise ImportError(
            f"{entry.module} requires the {entry.extra!r} extra. "
            f"Install with: pip install 'robotsix-llmio[{entry.extra}]'"
        ) from exc

    instance: LLMProvider = cls(**kwargs)
    return instance

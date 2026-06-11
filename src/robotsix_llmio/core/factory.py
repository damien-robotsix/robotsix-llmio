"""Provider-agnostic factory: select a backend by *name* + :class:`Tier`.

Consumers should obtain a provider through :func:`get_provider` and never
import a concrete provider class. The backend is resolved from config — an
explicit ``provider=`` argument, the ``LLMIO_PROVIDER`` environment variable,
or the built-in default (``"openrouter-deepseek"``) — so swapping the fleet to
another backend needs no consumer code change.

Provider classes are **lazy-imported** on resolution so optional extras stay
optional: a missing extra surfaces an :class:`ImportError` naming the extra and
the exact ``pip install`` command. Only ``importlib``, ``os``, typing, and the
``core.provider`` return type are imported at module load — never a heavy or
optional provider dependency.

Constructors are heterogeneous across backends, so :func:`get_provider`
forwards ``**kwargs`` to the resolved class rather than imposing a uniform
signature; the caller is responsible for passing kwargs the chosen backend
accepts.
"""

from __future__ import annotations

import importlib
import os
from dataclasses import dataclass
from typing import Any

from .provider import LLMProvider

_DEFAULT_PROVIDER = "openrouter-deepseek"
_ENV_VAR = "LLMIO_PROVIDER"


@dataclass(frozen=True)
class _ProviderEntry:
    """Lazy-import coordinates of a registered provider class."""

    module: str
    class_name: str
    extra: str


_PROVIDER_REGISTRY: dict[str, _ProviderEntry] = {
    "openrouter-deepseek": _ProviderEntry(
        module="robotsix_llmio.openrouter_deepseek",
        class_name="OpenRouterDeepseekProvider",
        extra="openrouter_deepseek",
    ),
    "claude-sdk": _ProviderEntry(
        module="robotsix_llmio.claude_sdk",
        class_name="ClaudeSDKProvider",
        extra="claude_sdk",
    ),
}


def register_provider(name: str, *, module: str, class_name: str, extra: str) -> None:
    """Register a provider *name* so consumers can select it by name.

    *module* is the dotted path of the provider sub-package, *class_name* the
    provider class attribute on it, and *extra* the pip extra used to build the
    actionable missing-extra error. Lets a new backend register its name once
    without editing the registry dict literal.
    """
    _PROVIDER_REGISTRY[name] = _ProviderEntry(
        module=module, class_name=class_name, extra=extra
    )


def get_provider(*, provider: str | None = None, **kwargs: Any) -> LLMProvider:
    """Resolve and instantiate a provider by registry *name*.

    The registry name is resolved in precedence order: an explicit *provider*
    argument, the ``LLMIO_PROVIDER`` environment variable, then the default
    ``"openrouter-deepseek"``. The provider class is lazy-imported so optional
    extras stay optional; ``**kwargs`` are forwarded to the resolved class
    constructor (the caller must pass kwargs the chosen backend accepts).
    """
    name = provider or os.environ.get(_ENV_VAR) or _DEFAULT_PROVIDER
    try:
        entry = _PROVIDER_REGISTRY[name]
    except KeyError as exc:
        known = ", ".join(sorted(_PROVIDER_REGISTRY))
        raise ValueError(
            f"Unknown provider {name!r}. Known providers: {known}."
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

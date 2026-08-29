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
from typing import TYPE_CHECKING, Any

from .provider import LLMProvider

if TYPE_CHECKING:
    from robotsix_llmio.config.tier import TierConfig


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
        module="robotsix_llmio.openrouter._deepseek_provider",
        class_name="OpenRouterDeepseekProvider",
        extra="openrouter",
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

    Args:
        identifier: Combined provider-model identifier — e.g.
            ``"claudeSDK-opus"`` or
            ``"openrouter-deepseek/deepseek-v4-flash-latest"``.
        **kwargs: Forwarded to the resolved provider class constructor.

    Returns:
        LLMProvider: A fully-instantiated provider.

    Raises:
        MalformedIdentifierError: If *identifier* cannot be parsed.
        ValueError: If the parsed provider prefix is not in
            :data:`_PROVIDER_PREFIX_MAP`.
        ImportError: If the provider's optional extra is not installed.

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


# ---------------------------------------------------------------------------
#  Level-based entry points — wire TierConfig into the factory
# ---------------------------------------------------------------------------


def default_tier_config() -> TierConfig:
    """Return a :class:`TierConfig` built from the baked per-level defaults.

    This is the single source of the baked per-level *(provider, model)*
    binding: level 1 → ``openrouter-deepseek/deepseek-v4-flash-20260731``,
    level 2 → ``claudeSDK-haiku``,
    level 3 → ``openrouter-xiaomi/mimo-v2.5-pro``,
    level 4 → ``claudeSDK-opus``,
    level 5 → ``claudeSDK-claude-fable-5``.

    The no-arg ``TierConfig()`` constructor uses ``default_factory``
    lambdas that produce independent deep copies of the module-level
    default singletons, so each call returns a fresh config whose slots
    do not alias the singletons or each other.

    Returns:
        TierConfig: A config whose five slots hold fresh copies of the
            baked defaults.

    """
    from robotsix_llmio.config.tier import TierConfig

    return TierConfig()


def get_provider_for_level(
    level: int,
    *,
    tier_config: TierConfig | None = None,
    **kwargs: Any,
) -> LLMProvider:
    """Resolve and instantiate the provider bound to a capability *level*.

    The *(provider, model)* binding for *level* is read from *tier_config*
    (or :func:`default_tier_config` when ``None``); the level's combined
    identifier drives the provider lazy-import, exactly as
    :func:`get_provider_for_identifier`.  The tier level's
    ``provider_kwargs`` are merged with ``**kwargs`` (caller ``**kwargs``
    win on conflict) and forwarded to the provider constructor.

    Args:
        level: Capability level — ``1`` (cheap), ``2`` (intermediate),
            ``3`` (high-level planning), or ``4`` (frontier).
        tier_config: Per-level *(provider, model)* binding to resolve
            against.  When ``None``, the baked defaults from
            :func:`default_tier_config` are used.
        **kwargs: Forwarded to the resolved provider class constructor,
            merged over the tier level's ``provider_kwargs`` (caller
            values win).

    Returns:
        LLMProvider: A fully-instantiated provider for the level's bound
            backend.

    Raises:
        ValueError: If *level* is not 1, 2, 3, 4, or 5 (via
            :meth:`TierConfig.for_level`), or if the level's identifier
            names an unknown provider prefix.
        MalformedIdentifierError: If the level's identifier cannot be
            parsed.
        ImportError: If the resolved provider's optional extra is not
            installed (e.g. ``claude_sdk`` for ``level=3`` or
            ``level=4``).

    """
    tlc = (tier_config or default_tier_config()).for_level(level)
    merged = {**tlc.provider_kwargs, **kwargs}
    if tlc.max_tokens is not None:
        merged.setdefault("max_tokens", tlc.max_tokens)
    return get_provider_for_identifier(tlc.model, **merged)


def build_agent_for_level(
    level: int,
    *,
    tier_config: TierConfig | None = None,
    model: str | None = None,
    provider_kwargs: dict[str, Any] | None = None,
    **build_kwargs: Any,
) -> Any:
    """Build an agent on the provider+model bound to a capability *level*.

    This is the single entry point a consumer needs: pick a *level* and get
    a ready-to-run agent on that level's default provider and model — no
    provider knowledge required.  Resolution is:

    1. resolve the level's :class:`~robotsix_llmio.config.tier.TierLevelConfig`
       from *tier_config* (or :func:`default_tier_config` when ``None``);
    2. lazy-import the provider via :func:`get_provider_for_identifier`,
       passing *provider_kwargs* (falling back to the level's
       ``provider_kwargs``) to its constructor;
    3. call ``provider.build_agent(level=level, model=..., **build_kwargs)``.

    With everything left at its default a consumer calls
    ``build_agent_for_level(1, system_prompt=..., output_type=str,
    name=...)`` for a cheap DeepSeek agent,
    ``build_agent_for_level(3, system_prompt=..., tools=...,
    output_type=str, name=...)`` for a Claude-opus agent, and
    ``build_agent_for_level(4, system_prompt=..., name=...)`` for a
    Claude-Fable-5 frontier agent.

    Args:
        level: Capability level — ``1`` (cheap), ``2`` (intermediate),
            ``3`` (high-level planning), or ``4`` (frontier).
        tier_config: Per-level *(provider, model)* binding to resolve
            against.  When ``None``, the baked defaults from
            :func:`default_tier_config` are used (i.e. omitting
            everything uses the baked defaults).
        model: Optional **bare** model-name override.  It overrides
            *only* the model name; the provider stays the one bound to
            *level*.  When ``None`` (default), the level's baked model
            name is used.
        provider_kwargs: Keyword arguments for the provider constructor.
            When ``None``, the level's ``provider_kwargs`` are used.
        **build_kwargs: Forwarded verbatim to ``provider.build_agent``
            (e.g. ``system_prompt``, ``tools``, ``output_type``,
            ``name``, ``retries``, ``builtin_tools``).

    Returns:
        AgentHandle: A ready-to-run agent handle.  Call ``.close()``
            when done.

    Raises:
        ValueError: If *level* is not 1, 2, 3, 4, or 5 (via
            :meth:`TierConfig.for_level`), or if the level's identifier
            names an unknown provider prefix.
        MalformedIdentifierError: If the level's identifier cannot be
            parsed.
        ImportError: If the resolved provider's optional extra is not
            installed (``build_agent_for_level(3)`` and
            ``build_agent_for_level(4)`` require the ``claude_sdk``
            extra).

    """
    tlc = (tier_config or default_tier_config()).for_level(level)
    kwargs = provider_kwargs if provider_kwargs is not None else tlc.provider_kwargs
    merged = dict(kwargs) if kwargs else {}
    if tlc.max_tokens is not None:
        merged.setdefault("max_tokens", tlc.max_tokens)
    provider = get_provider_for_identifier(tlc.model, **merged)
    return provider.build_agent(
        level=level, model=model or tlc.model_name, **build_kwargs
    )

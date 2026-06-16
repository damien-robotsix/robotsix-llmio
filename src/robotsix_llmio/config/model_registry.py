"""Per-provider named-model registry for early validation of model names.

This module provides a central registry of which model names each transport
provider can serve, plus a validation function that cross-checks model names
at :class:`TierLevelConfig` construction time.

The registry **duplicates** knowledge already present in provider modules
(constructor defaults / tier→model maps) so that misconfigured model names
are caught at config parse time rather than deep inside ``new_model()``.
"""

from __future__ import annotations

from robotsix_llmio.exceptions import RobotsixLLMIOError

# --------------------------------------------------------------------------- #
#  Registry                                                                   #
# --------------------------------------------------------------------------- #

PROVIDER_MODELS: dict[str, frozenset[str]] = {
    "openrouter-deepseek": frozenset(
        {
            "deepseek/deepseek-v4-pro",
            "deepseek/deepseek-v4-flash",
        }
    ),
    "claude-sdk": frozenset(
        {
            "opus",
            "haiku",
            "sonnet",
        }
    ),
}
"""Known provider → set-of-model-names mapping.

Keys match :data:`~robotsix_llmio.core.factory._PROVIDER_REGISTRY` keys.
Values are :class:`frozenset`\\ s to signal immutability of this module-level
constant.
"""


# --------------------------------------------------------------------------- #
#  Exception                                                                  #
# --------------------------------------------------------------------------- #


class UnknownModelError(RobotsixLLMIOError):
    """Raised when a model name is not known for a given provider."""


# --------------------------------------------------------------------------- #
#  Validation                                                                 #
# --------------------------------------------------------------------------- #


def validate_model(provider: str, model: str) -> None:
    """Check that *model* is a known model name for *provider*.

    Args:
        provider: Provider registry name (e.g. ``"openrouter-deepseek"``).
        model: Model name to validate.

    Returns:
        ``None`` if the model is valid or the provider is unknown.

    Raises:
        UnknownModelError: If *provider* is known but *model* is not in its
            set of known models.  The message names the provider, the
            unknown model, and the known models.
    """
    known_models = PROVIDER_MODELS.get(provider)
    if known_models is None:
        return  # unknown provider — validated elsewhere (by get_provider)
    if model not in known_models:
        raise UnknownModelError(
            f"Unknown model {model!r} for provider {provider!r}. "
            f"Known models: {', '.join(sorted(known_models))}."
        )

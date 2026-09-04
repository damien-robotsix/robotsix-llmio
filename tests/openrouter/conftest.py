"""Skip only the OpenRouter modules that import the transport/provider when
the optional ``openai>=3`` extra is missing or too old.

The OpenRouter transport layer imports ``openai`` eagerly (``model.py``:
``from openai import AsyncStream``) and constructs ``AsyncOpenAI`` with the
httpx2 client returned by ``core.timeout_http_client()`` — a client that only
``openai>=3`` accepts (``openai<3`` raises "Expected an instance of
httpx.AsyncClient but got httpx2.AsyncClient"). So when the optional ``openai``
extra is absent (the base env the implement gate runs) or a stale ``openai<3``
is installed, collecting or running a module that pulls that layer raises
instead of skipping.

Centralising the guard here means no individual ``tests/openrouter/`` module
needs its own ``pytest.importorskip("openai", minversion="3")`` guard. The
ignore list is scoped to the modules that actually import the transport/
provider layer: purely offline modules (``test_price_ceiling_check.py``,
``test_async_client.py``, ``test_openrouter_provider_cost.py``) stay collected
and runnable in the base env, so real regressions in that pure logic still
fail the gate. ``test_openrouter_deepseek.py`` is covered too: some of its
tests import the transport layer directly before their ``pydantic_ai``
importorskip can run, so without the guard they would ERROR rather than skip.
When the pinned extra is present every module collects and runs as intended.
"""

from __future__ import annotations

from importlib.metadata import PackageNotFoundError, version


def _openai_v3_available() -> bool:
    """True only when ``openai>=3`` is installed (metadata-only check).

    Reads distribution metadata rather than importing ``openai`` so the check
    works even in the base env where the import itself would fail.
    """
    try:
        raw = version("openai")
    except PackageNotFoundError:
        return False
    try:
        major = int(raw.split(".", 1)[0])
    except ValueError:
        return False
    return major >= 3


# Modules whose collection (or pre-skip test-body import) pulls the
# transport/provider layer, which does a module-level ``from openai import
# AsyncStream`` (``src/robotsix_llmio/openrouter/model.py``). Only these break
# in the base env; the offline modules above must keep running without the
# extra.
_OPENAI_IMPORTING_MODULES: tuple[str, ...] = (
    "test_openrouter.py",
    "test_openrouter_provider.py",
    "test_openrouter_deepseek.py",
    "test_openrouter_deepseek_live.py",
)

# When ``openai>=3`` is unavailable, ignore just the transport/provider
# modules so collection never touches the eager ``from openai import ...``
# nor instantiates a provider against a stale ``openai<3``.
collect_ignore_glob: list[str] = (
    [] if _openai_v3_available() else list(_OPENAI_IMPORTING_MODULES)
)

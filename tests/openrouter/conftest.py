"""Skip the whole OpenRouter suite when the optional ``openai>=3`` extra
is missing or too old.

The OpenRouter transport layer imports ``openai`` eagerly (``model.py``:
``from openai import AsyncStream``) and constructs ``AsyncOpenAI`` with the
httpx2 client returned by ``core.timeout_http_client()`` — a client that only
``openai>=3`` accepts (``openai<3`` raises "Expected an instance of
httpx.AsyncClient but got httpx2.AsyncClient"). So when the optional ``openai``
extra is absent (the base env the implement gate runs) or a stale ``openai<3``
is installed, collecting or running these modules raises instead of skipping.

Centralising the guard here means no individual ``tests/openrouter/`` module
needs its own ``pytest.importorskip("openai", minversion="3")`` guard: when
``openai>=3`` is unavailable pytest silently ignores the whole directory, so
``pytest tests/openrouter/`` reports no ERROR. When the pinned extra is present
every module collects and runs as intended.
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


# When ``openai>=3`` is unavailable, ignore every module in this directory so
# collection never touches the eager ``from openai import ...`` in the
# transport layer nor instantiates a provider against a stale ``openai<3``.
collect_ignore_glob: list[str] = [] if _openai_v3_available() else ["*.py"]

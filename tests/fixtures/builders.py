"""``SimpleNamespace`` / mock-object builders for the test suite.

These centralize the inline ``SimpleNamespace`` constructions used to fake
the pydantic-ai response/usage objects that the cost helpers read. The
canonical example is OpenRouter cost extraction, which inspects
``response.usage.cost`` and ``response.usage.model_extra``.

Prefer these builders over hand-rolling ``SimpleNamespace(...)`` inline so
the faked shape stays consistent with the attributes production code reads.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

# Sentinel so callers can distinguish "attribute omitted entirely" from
# "attribute explicitly set to None". ``make_openrouter_response(usage=None)``
# must model a response with no usage block, which is a real code path.
_UNSET: Any = object()


def make_openrouter_usage(
    *, cost: Any = _UNSET, model_extra: Any = _UNSET
) -> SimpleNamespace:
    """Build a fake OpenRouter ``usage`` namespace.

    Only the attributes you pass are present; omit ``cost`` or
    ``model_extra`` to leave that attribute off the namespace entirely
    (mirroring responses where the field is absent rather than ``None``).

    >>> make_openrouter_usage(cost=0.0123, model_extra=None).cost
    0.0123
    >>> hasattr(make_openrouter_usage(model_extra={"cost": 0.5}), "cost")
    False
    """
    attrs: dict[str, Any] = {}
    if cost is not _UNSET:
        attrs["cost"] = cost
    if model_extra is not _UNSET:
        attrs["model_extra"] = model_extra
    return SimpleNamespace(**attrs)


def make_openrouter_response(
    *, usage: Any = _UNSET, cost: Any = _UNSET, model_extra: Any = _UNSET
) -> SimpleNamespace:
    """Build a fake OpenRouter response object for cost-extraction tests.

    Mirrors the pydantic-ai response shape the cost helper reads
    (``response.usage.cost`` / ``response.usage.model_extra``):

    * Pass ``usage=None`` (or any ready-made object) to set ``response.usage``
      directly — use this to model a response with no usage block.
    * Otherwise ``cost`` / ``model_extra`` populate a usage namespace built
      via :func:`make_openrouter_usage`; omit either to leave that attribute
      off entirely.

    >>> make_openrouter_response(cost=0.0123).usage.cost
    0.0123
    >>> make_openrouter_response(usage=None).usage is None
    True
    """
    if usage is not _UNSET:
        return SimpleNamespace(usage=usage)
    return SimpleNamespace(
        usage=make_openrouter_usage(cost=cost, model_extra=model_extra)
    )

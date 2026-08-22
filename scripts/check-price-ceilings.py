#!/usr/bin/env python3
"""Drift guard for the baked DeepSeek-on-OpenRouter price ceilings.

``_deepseek_model.py`` bakes ``max_price`` ceilings (and a short provider
``ignore`` list for cache-read outliers) that were measured against
OpenRouter's live per-provider endpoint list. Nothing enforces those ceilings:
when DeepSeek or its neighbours reprice, the ceiling silently stops admitting
endpoints — ``404 No endpoints found …`` on the capable tier, or the preferred
provider silently dropping out on the cheap tier — and nothing in the fleet
notices until a human goes looking.

This script is that guard. For each DeepSeek model in the baked tier defaults
it fetches ``GET /api/v1/models/{model}/endpoints`` and checks, against the
ceilings the library actually ships:

1. the preferred provider (``DeepSeek``) still satisfies its tier ceiling;
   a missing preferred provider is a warning, not a failure;
2. at least ``--min-healthy`` healthy endpoints satisfy the ceiling;
3. no admitted (ceiling-passing, non-ignored) endpoint's ``input_cache_read``
   exceeds the preferred provider's by more than ``--max-cache-read-factor``.

On any failure it prints the measured per-provider table and exits non-zero so
a scheduled run fails loudly. It is deliberately NOT a per-PR CI gate — an
upstream price move must not block unrelated merges. Run it::

    python scripts/check-price-ceilings.py
"""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import httpx

from robotsix_llmio.config.tier import LEVEL1_DEFAULT, LEVEL2_DEFAULT
from robotsix_llmio.core.constants import HTTP_CLIENT_TIMEOUT
from robotsix_llmio.openrouter._base import _DEFAULT_BASE_URL
from robotsix_llmio.openrouter._deepseek_model import (
    _PREFERRED_PROVIDER,
    DEFAULT_IGNORE_CAPABLE,
    DEFAULT_MAX_PRICE_CAPABLE,
    DEFAULT_MAX_PRICE_CHEAP,
)

# OpenRouter reports endpoint prices as USD per TOKEN (string); the library's
# ceilings are USD per 1M tokens.
_TOKENS_PER_MILLION = 1_000_000

# Defaults for the three assertions. ``_CACHE_READ_MAX_FACTOR`` is a
# conservative multiple: it should flag providers whose cache-read rate is
# several times DeepSeek's (the ``provider.ignore`` list exists for exactly
# those) while tolerating the small spread among ordinary providers.
_MIN_HEALTHY_ENDPOINTS = 3
_CACHE_READ_MAX_FACTOR = 4.0


@dataclass(frozen=True)
class EndpointSnapshot:
    """One endpoint's prices, normalised to USD per 1M tokens."""

    provider: str
    status: int | None
    prompt: float
    completion: float
    cache_read: float


@dataclass(frozen=True)
class TierCheck:
    """A baked tier's model + ceiling + ignore list, sourced from the library."""

    label: str
    model: str
    ceiling: dict[str, float]
    ignore: tuple[str, ...]
    preferred_provider: str = "DeepSeek"


@dataclass
class TierReport:
    """Result of checking one tier, including the measured endpoint table.

    ``admitted_indices`` indexes into ``endpoints`` the subset that would
    actually be routed to (within the ceiling and not ignored).
    """

    check: TierCheck
    endpoints: list[EndpointSnapshot]
    admitted_indices: set[int]
    failures: list[str]
    warnings: list[str]
    preferred_admitted: bool = False
    reference_cache_read: float | None = None


def _ceiling_from_kwargs(kwargs: dict[str, Any]) -> dict[str, float] | None:
    """Extract a ``max_price`` ceiling from *provider_kwargs* if present."""
    prompt = kwargs.get("max_price_prompt")
    completion = kwargs.get("max_price_completion")
    if prompt is not None and completion is not None:
        return {"prompt": float(prompt), "completion": float(completion)}
    return None


def _ignore_from_kwargs(kwargs: dict[str, Any]) -> tuple[str, ...] | None:
    """Extract ``ignore_providers`` from *provider_kwargs* if present."""
    ignore = kwargs.get("ignore_providers")
    if ignore is not None:
        return tuple(ignore)
    return None


def _preferred_from_kwargs(kwargs: dict[str, Any]) -> str:
    """Extract the preferred provider from *provider_kwargs*, defaulting to
    DeepSeek."""
    return str(kwargs.get("preferred_provider", _PREFERRED_PROVIDER))


#: The OpenRouter tiers from the baked tier defaults. Ceilings, ignore lists,
#: and preferred providers come from each tier's ``provider_kwargs`` where
#: present, falling back to the module-level DeepSeek defaults — so the guard
#: covers whatever model the tier currently binds rather than hard-coding
#: DeepSeek.
_TIERS: tuple[TierCheck, ...] = (
    TierCheck(
        label="cheap (level 1)",
        model=LEVEL1_DEFAULT.model_name,
        ceiling=_ceiling_from_kwargs(LEVEL1_DEFAULT.provider_kwargs)
        or DEFAULT_MAX_PRICE_CHEAP,
        ignore=_ignore_from_kwargs(LEVEL1_DEFAULT.provider_kwargs) or (),
        preferred_provider=_preferred_from_kwargs(LEVEL1_DEFAULT.provider_kwargs),
    ),
    TierCheck(
        label="capable (level 2)",
        model=LEVEL2_DEFAULT.model_name,
        ceiling=_ceiling_from_kwargs(LEVEL2_DEFAULT.provider_kwargs)
        or DEFAULT_MAX_PRICE_CAPABLE,
        ignore=_ignore_from_kwargs(LEVEL2_DEFAULT.provider_kwargs)
        or DEFAULT_IGNORE_CAPABLE,
        preferred_provider=_preferred_from_kwargs(LEVEL2_DEFAULT.provider_kwargs),
    ),
)


def _price_per_million(raw: Any) -> float:
    """Normalise an OpenRouter per-token USD price to USD per 1M tokens.

    ``None`` (a dimension the endpoint does not price) normalises to zero.
    """
    if raw is None:
        return 0.0
    return float(raw) * _TOKENS_PER_MILLION


def _snapshot(endpoint: dict[str, Any]) -> EndpointSnapshot:
    """Extract the prices a ceiling/guard cares about from one endpoint."""
    pricing = endpoint.get("pricing") or {}
    return EndpointSnapshot(
        provider=str(endpoint.get("provider_name") or "unknown"),
        status=endpoint.get("status"),
        prompt=_price_per_million(pricing.get("prompt")),
        completion=_price_per_million(pricing.get("completion")),
        cache_read=_price_per_million(pricing.get("input_cache_read")),
    )


def _satisfies(snapshot: EndpointSnapshot, ceiling: dict[str, float]) -> bool:
    """Whether *snapshot*'s prompt/completion both fit under *ceiling*."""
    return (
        snapshot.prompt <= ceiling["prompt"]
        and snapshot.completion <= ceiling["completion"]
    )


def _is_healthy(status: int | None) -> bool:
    """An endpoint is healthy when its status is ``None`` (absent) or >= 0.

    OpenRouter's endpoint ``status`` is an integer: ``0`` for a normal
    endpoint, negative values for degraded/disabled.
    """
    return status is None or status >= 0


def check_tier(
    check: TierCheck,
    endpoints: Sequence[dict[str, Any]],
    *,
    preferred_provider: str | None = None,
    min_healthy: int = _MIN_HEALTHY_ENDPOINTS,
    max_cache_read_factor: float = _CACHE_READ_MAX_FACTOR,
) -> TierReport:
    """Check one tier's live endpoints against its baked ceiling/ignore list.

    Returns a :class:`TierReport` whose ``failures`` is empty when the tier is
    healthy; otherwise it holds one human-readable line per violated assertion.
    """
    if preferred_provider is None:
        preferred_provider = check.preferred_provider
    snapshots = [_snapshot(ep) for ep in endpoints]
    ceiling = check.ceiling
    admitted_indices = {
        i
        for i, snapshot in enumerate(snapshots)
        if _satisfies(snapshot, ceiling) and snapshot.provider not in check.ignore
    }
    admitted = [snapshots[i] for i in sorted(admitted_indices)]
    report = TierReport(
        check=check,
        endpoints=snapshots,
        admitted_indices=admitted_indices,
        failures=[],
        warnings=[],
    )

    preferred = [s for s in snapshots if s.provider == preferred_provider]
    if not preferred:
        report.warnings.append(
            f"{check.label}: preferred provider {preferred_provider!r} has no "
            f"endpoints for {check.model}"
        )
    else:
        report.preferred_admitted = any(_satisfies(s, ceiling) for s in preferred)
        if not report.preferred_admitted:
            cheapest = min(preferred, key=lambda s: (s.prompt, s.completion))
            report.failures.append(
                f"{check.label}: preferred provider {preferred_provider!r} lists "
                f"${cheapest.prompt:.3f}/${cheapest.completion:.3f}, above the "
                f"${ceiling['prompt']:.2f}/${ceiling['completion']:.2f} ceiling"
            )

    healthy = [s for s in admitted if _is_healthy(s.status)]
    if len(healthy) < min_healthy:
        report.failures.append(
            f"{check.label}: only {len(healthy)} healthy endpoint(s) satisfy "
            f"the ceiling (need {min_healthy})"
        )

    report.reference_cache_read = (
        min((s.cache_read for s in preferred), default=None) if preferred else None
    )
    if report.reference_cache_read and report.reference_cache_read > 0:
        for s in admitted:
            if s.provider == preferred_provider:
                continue
            if s.cache_read > report.reference_cache_read * max_cache_read_factor:
                report.failures.append(
                    f"{check.label}: {s.provider} cache-read "
                    f"${s.cache_read:.3f}/M exceeds {preferred_provider} "
                    f"${report.reference_cache_read:.3f}/M by more than "
                    f"{max_cache_read_factor:g}x"
                )

    return report


def fetch_endpoints(
    client: httpx.Client,
    model: str,
    *,
    base_url: str = _DEFAULT_BASE_URL,
) -> list[dict[str, Any]]:
    """Fetch the live endpoint list for *model* from OpenRouter.

    Raises :class:`httpx.HTTPError` on transport or HTTP failure so a scheduled
    run fails loudly rather than reading an absent list as "all clear".
    """
    url = f"{base_url.rstrip('/')}/models/{model}/endpoints"
    response = client.get(url)
    response.raise_for_status()
    payload = response.json()
    data = payload.get("data") if isinstance(payload, dict) else None
    endpoints = data.get("endpoints") if isinstance(data, dict) else None
    return list(endpoints or [])


def _render_tier(report: TierReport) -> str:
    """Render one tier's measured table plus any failures as Markdown."""
    ceiling = report.check.ceiling
    lines = [
        f"## {report.check.label} — {report.check.model}",
        "",
        f"Ceiling: prompt <= ${ceiling['prompt']:.2f}/M, "
        f"completion <= ${ceiling['completion']:.2f}/M; "
        f"ignore: {', '.join(report.check.ignore) or '—'}",
        "",
        "| provider | status | prompt $/M | completion $/M | cache-read $/M | "
        "admitted |",
        "|---|---|---|---|---|---|",
    ]
    for index, s in enumerate(report.endpoints):
        admitted = "yes" if index in report.admitted_indices else "no"
        lines.append(
            f"| {s.provider} | {s.status if s.status is not None else '—'} | "
            f"{s.prompt:.3f} | {s.completion:.3f} | {s.cache_read:.3f} | "
            f"{admitted} |"
        )
    if report.failures:
        lines.append("")
        lines.append("**Failures:**")
        lines.extend(f"- {failure}" for failure in report.failures)
    if report.warnings:
        lines.append("")
        lines.append("**Warnings:**")
        lines.extend(f"- {warning}" for warning in report.warnings)
    if not report.failures and not report.warnings:
        lines.append("")
        lines.append("**OK**")
    return "\n".join(lines)


def _parse_args(argv: Sequence[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Check the baked DeepSeek price ceilings against "
        "OpenRouter's live per-provider endpoint list."
    )
    parser.add_argument(
        "--base-url",
        default=_DEFAULT_BASE_URL,
        help="OpenRouter API base URL (default: %(default)s).",
    )
    parser.add_argument(
        "--min-healthy",
        type=int,
        default=_MIN_HEALTHY_ENDPOINTS,
        help="Minimum healthy endpoints that must satisfy each ceiling "
        "(default: %(default)s).",
    )
    parser.add_argument(
        "--max-cache-read-factor",
        type=float,
        default=_CACHE_READ_MAX_FACTOR,
        help="Maximum admitted cache-read multiple of the preferred provider's "
        "(default: %(default)s).",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    """Run the guard over every baked DeepSeek tier; return the exit code."""
    args = _parse_args(argv)
    reports: list[TierReport] = []
    with httpx.Client(timeout=HTTP_CLIENT_TIMEOUT) as client:
        for check in _TIERS:
            endpoints = fetch_endpoints(client, check.model, base_url=args.base_url)
            reports.append(
                check_tier(
                    check,
                    endpoints,
                    min_healthy=args.min_healthy,
                    max_cache_read_factor=args.max_cache_read_factor,
                )
            )

    print("\n\n".join(_render_tier(report) for report in reports))
    total_failures = sum(len(report.failures) for report in reports)
    total_warnings = sum(len(report.warnings) for report in reports)
    print()
    if total_failures:
        msg = f"PRICE CEILING DRIFT DETECTED: {total_failures} violation(s)."
        if total_warnings:
            msg += f" ({total_warnings} additional warning(s) — see above.)"
        print(msg)
        return 1
    if total_warnings:
        print(
            f"All DeepSeek price ceilings verified. "
            f"{total_warnings} warning(s) — see above."
        )
        return 0
    print("All DeepSeek price ceilings verified against the live endpoint list.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

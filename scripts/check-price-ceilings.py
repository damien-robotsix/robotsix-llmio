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

Rule 3 has a deterministic remedy — bar the outlier via the tier's ignore
list — and third-party repricing trips it most weeks, so ``--fix`` applies
that remedy directly to the library source (the ``ignore_providers`` list of
``FALLBACK_LEVEL1``/``FALLBACK_LEVEL2`` in ``config/tier.py``, the
``DEFAULT_IGNORE_CAPABLE`` tuple in ``openrouter/_deepseek_model.py``) and
re-checks the tier with the extended list. The scheduled workflow runs it on a
failure and opens a PR with the diff; rules 1 and 2 (a ceiling that no longer
admits the preferred provider or fewer than three healthy endpoints) have no
mechanical fix and still fail the run. ``--summary-file`` writes the Markdown
report (tables + applied fixes) for the PR body.
"""

from __future__ import annotations

import argparse
import re
from collections.abc import Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import httpx

from robotsix_llmio.config.tier import (
    FALLBACK_LEVEL1,
    FALLBACK_LEVEL2,
    FALLBACK_LEVEL3,
)
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
    #: Where ``--fix`` appends cache-read outliers: ``(relative source path,
    #: anchor)``. The anchor is the ``FALLBACK_LEVELn`` block whose
    #: ``provider_kwargs["ignore_providers"]`` list is extended, or the
    #: module-level tuple name (``DEFAULT_IGNORE_CAPABLE``). ``None`` = not
    #: auto-fixable.
    ignore_source: tuple[str, str] | None = None


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
    #: Rule-3 offenders as ``(provider, cache_read $/M)``, deduplicated —
    #: the input of ``--fix``.
    cache_read_outliers: list[tuple[str, float]] = field(default_factory=list)


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


#: Library files ``--fix`` edits, relative to the repository root.
_TIER_SOURCE = "src/robotsix_llmio/config/tier.py"
_DEEPSEEK_SOURCE = "src/robotsix_llmio/openrouter/_deepseek_model.py"

#: The OpenRouter tiers from the baked tier defaults. Ceilings, ignore lists,
#: and preferred providers come from each tier's ``provider_kwargs`` where
#: present, falling back to the module-level DeepSeek defaults — so the guard
#: covers whatever model the tier currently binds rather than hard-coding
#: DeepSeek.
_TIERS: tuple[TierCheck, ...] = (
    TierCheck(
        label="cheap (level 1)",
        model=FALLBACK_LEVEL1.model_name,
        ceiling=_ceiling_from_kwargs(FALLBACK_LEVEL1.provider_kwargs)
        or DEFAULT_MAX_PRICE_CHEAP,
        ignore=_ignore_from_kwargs(FALLBACK_LEVEL1.provider_kwargs) or (),
        preferred_provider=_preferred_from_kwargs(FALLBACK_LEVEL1.provider_kwargs),
        ignore_source=(_TIER_SOURCE, "FALLBACK_LEVEL1"),
    ),
    TierCheck(
        label="cheap-reasoning (level 2)",
        model=FALLBACK_LEVEL2.model_name,
        ceiling=_ceiling_from_kwargs(FALLBACK_LEVEL2.provider_kwargs)
        or DEFAULT_MAX_PRICE_CHEAP,
        ignore=_ignore_from_kwargs(FALLBACK_LEVEL2.provider_kwargs) or (),
        preferred_provider=_preferred_from_kwargs(FALLBACK_LEVEL2.provider_kwargs),
        ignore_source=(_TIER_SOURCE, "FALLBACK_LEVEL2"),
    ),
    TierCheck(
        label="capable (level 3)",
        model=FALLBACK_LEVEL3.model_name,
        ceiling=_ceiling_from_kwargs(FALLBACK_LEVEL3.provider_kwargs)
        or DEFAULT_MAX_PRICE_CAPABLE,
        ignore=_ignore_from_kwargs(FALLBACK_LEVEL3.provider_kwargs)
        or DEFAULT_IGNORE_CAPABLE,
        preferred_provider=_preferred_from_kwargs(FALLBACK_LEVEL3.provider_kwargs),
        ignore_source=(
            (_TIER_SOURCE, "FALLBACK_LEVEL3")
            if _ignore_from_kwargs(FALLBACK_LEVEL3.provider_kwargs) is not None
            else (_DEEPSEEK_SOURCE, "DEFAULT_IGNORE_CAPABLE")
        ),
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
                if s.provider not in {name for name, _ in report.cache_read_outliers}:
                    report.cache_read_outliers.append((s.provider, s.cache_read))

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


def _ignore_entry_lines(
    providers: Sequence[tuple[str, float]],
    *,
    indent: str,
    preferred_provider: str,
    reference_cache_read: float | None,
    today: str,
) -> list[str]:
    """Render ``"Provider",`` entries, each preceded by a dated audit comment."""
    lines: list[str] = []
    for provider, cache_read in providers:
        reference = (
            f" vs {preferred_provider} ${reference_cache_read:.3f}/M"
            if reference_cache_read
            else ""
        )
        lines.append(
            f"{indent}# {today} drift-fix: cache-read ${cache_read:.3f}/M{reference}"
        )
        lines.append(f'{indent}"{provider}",')
    return lines


def extend_ignore_list_in_block(
    text: str,
    block: str,
    providers: Sequence[tuple[str, float]],
    *,
    preferred_provider: str,
    reference_cache_read: float | None,
    today: str,
) -> str:
    """Append *providers* to ``"ignore_providers": [...]`` inside the
    ``<block> = TierLevelConfig(`` literal of ``config/tier.py``.

    When the block has no ``ignore_providers`` key yet, one is inserted right
    after its ``"preferred_provider"`` line. Raises :class:`ValueError` when
    the block or the insertion point cannot be found — the source layout the
    fixer relies on has changed and a human must look.
    """
    start_match = re.search(rf"^{re.escape(block)} = TierLevelConfig\(\n", text, re.M)
    if start_match is None:
        raise ValueError(f"{block} = TierLevelConfig( not found")
    end = text.find("\n)\n", start_match.end())
    if end == -1:
        raise ValueError(f"{block}: closing paren not found")
    body = text[start_match.end() : end]
    list_match = re.search(
        r'^(\s*)"ignore_providers": \[(.*?)\],?\n', body, re.S | re.M
    )
    if list_match is not None:
        indent = list_match.group(1)
        inner = list_match.group(2).strip()
        existing = (
            [e.strip().strip("\"'") for e in inner.split(",") if e.strip()]
            if inner and "#" not in inner
            else []
        )
        if not existing and inner:
            existing = re.findall(r"[\"']([^\"']+)[\"']", inner)
        new = [(p, c) for p, c in providers if p not in existing]
        if not new:
            return text
        entry_indent = indent + "    "
        entries = [f'{entry_indent}"{e}",' for e in existing] + _ignore_entry_lines(
            new,
            indent=entry_indent,
            preferred_provider=preferred_provider,
            reference_cache_read=reference_cache_read,
            today=today,
        )
        replacement = (
            f'{indent}"ignore_providers": [\n' + "\n".join(entries) + f"\n{indent}],\n"
        )
        new_body = body[: list_match.start()] + replacement + body[list_match.end() :]
    else:
        pref_match = re.search(r'^(\s*)"preferred_provider": .*\n', body, re.M)
        if pref_match is None:
            raise ValueError(f"{block}: no ignore_providers and no preferred_provider")
        indent = pref_match.group(1)
        entries = _ignore_entry_lines(
            providers,
            indent=indent + "    ",
            preferred_provider=preferred_provider,
            reference_cache_read=reference_cache_read,
            today=today,
        )
        insertion = (
            f'{indent}"ignore_providers": [\n' + "\n".join(entries) + f"\n{indent}],\n"
        )
        new_body = body[: pref_match.end()] + insertion + body[pref_match.end() :]
    return text[: start_match.end()] + new_body + text[end:]


def extend_ignore_tuple(
    text: str,
    name: str,
    providers: Sequence[tuple[str, float]],
    *,
    preferred_provider: str,
    reference_cache_read: float | None,
    today: str,
) -> str:
    """Append *providers* to the module-level ``<name>: tuple[str, ...] = (``
    literal (``DEFAULT_IGNORE_CAPABLE`` in ``_deepseek_model.py``)."""
    match = re.search(
        rf"^{re.escape(name)}: tuple\[str, \.\.\.\] = \(\n(.*?)^\)\n", text, re.S | re.M
    )
    if match is None:
        raise ValueError(f"{name} tuple literal not found")
    existing = re.findall(r"^\s*[\"']([^\"']+)[\"'],", match.group(1), re.M)
    new = [(p, c) for p, c in providers if p not in existing]
    if not new:
        return text
    entries = _ignore_entry_lines(
        new,
        indent="    ",
        preferred_provider=preferred_provider,
        reference_cache_read=reference_cache_read,
        today=today,
    )
    body = match.group(1).rstrip("\n") + "\n" + "\n".join(entries) + "\n"
    return text[: match.start(1)] + body + text[match.end(1) :]


def apply_cache_read_fixes(
    reports: Sequence[TierReport],
    *,
    repo_root: Path,
    today: str | None = None,
) -> list[str]:
    """Bar every tier's rule-3 outliers in the library source.

    Returns one human-readable line per applied edit (empty when nothing was
    fixable). Tiers without an ``ignore_source`` are left alone; a source
    layout the fixer does not recognise raises :class:`ValueError`.
    """
    today = today or datetime.now(UTC).date().isoformat()
    applied: list[str] = []
    for report in reports:
        source = report.check.ignore_source
        if not report.cache_read_outliers or source is None:
            continue
        rel_path, anchor = source
        path = repo_root / rel_path
        text = path.read_text(encoding="utf-8")
        kwargs: dict[str, Any] = {
            "preferred_provider": report.check.preferred_provider,
            "reference_cache_read": report.reference_cache_read,
            "today": today,
        }
        if anchor.startswith("FALLBACK_LEVEL"):
            updated = extend_ignore_list_in_block(
                text, anchor, report.cache_read_outliers, **kwargs
            )
        else:
            updated = extend_ignore_tuple(
                text, anchor, report.cache_read_outliers, **kwargs
            )
        if updated == text:
            continue
        path.write_text(updated, encoding="utf-8")
        names = ", ".join(name for name, _ in report.cache_read_outliers)
        applied.append(f"{report.check.label}: barred {names} in {rel_path} ({anchor})")
    return applied


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
        "--fix",
        action="store_true",
        help="Bar cache-read outliers (rule 3) in the library source and "
        "re-check with the extended ignore lists; exit non-zero only for "
        "failures that have no mechanical remedy.",
    )
    parser.add_argument(
        "--repo-root",
        type=Path,
        default=Path(__file__).resolve().parents[1],
        help="Repository root whose src/ --fix edits (default: this checkout).",
    )
    parser.add_argument(
        "--summary-file",
        type=Path,
        default=None,
        help="Also write the Markdown report (tables, failures, applied fixes) "
        "to this file, e.g. for a PR body.",
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
    endpoints_by_tier: list[list[dict[str, Any]]] = []
    with httpx.Client(timeout=HTTP_CLIENT_TIMEOUT) as client:
        for check in _TIERS:
            endpoints = fetch_endpoints(client, check.model, base_url=args.base_url)
            endpoints_by_tier.append(endpoints)
            reports.append(
                check_tier(
                    check,
                    endpoints,
                    min_healthy=args.min_healthy,
                    max_cache_read_factor=args.max_cache_read_factor,
                )
            )

    sections = [_render_tier(report) for report in reports]
    applied: list[str] = []
    if args.fix and any(report.cache_read_outliers for report in reports):
        applied = apply_cache_read_fixes(reports, repo_root=args.repo_root)
        # Re-check every fixed tier with its extended ignore list so the exit
        # code reflects what the PR would ship, not the pre-fix state.
        rechecked: list[TierReport] = []
        for report, endpoints in zip(reports, endpoints_by_tier, strict=True):
            if not report.cache_read_outliers:
                rechecked.append(report)
                continue
            extended = TierCheck(
                label=report.check.label,
                model=report.check.model,
                ceiling=report.check.ceiling,
                ignore=report.check.ignore
                + tuple(name for name, _ in report.cache_read_outliers),
                preferred_provider=report.check.preferred_provider,
                ignore_source=report.check.ignore_source,
            )
            rechecked.append(
                check_tier(
                    extended,
                    endpoints,
                    min_healthy=args.min_healthy,
                    max_cache_read_factor=args.max_cache_read_factor,
                )
            )
        reports = rechecked
        sections.append(
            "## Applied fixes (--fix)\n\n"
            + ("\n".join(f"- {line}" for line in applied) or "- none")
            + "\n\n## After fix\n\n"
            + "\n\n".join(_render_tier(report) for report in reports)
        )

    output = "\n\n".join(sections)
    print(output)
    total_failures = sum(len(report.failures) for report in reports)
    total_warnings = sum(len(report.warnings) for report in reports)
    print()
    if total_failures:
        msg = f"PRICE CEILING DRIFT DETECTED: {total_failures} violation(s)."
        if total_warnings:
            msg += f" ({total_warnings} additional warning(s) — see above.)"
        if applied:
            msg += " Remaining after --fix — no mechanical remedy."
        print(msg)
        verdict = msg
        code = 1
    elif total_warnings:
        verdict = (
            f"All DeepSeek price ceilings verified. "
            f"{total_warnings} warning(s) — see above."
        )
        print(verdict)
        code = 0
    else:
        verdict = "All DeepSeek price ceilings verified against the live endpoint list."
        print(verdict)
        code = 0
    if args.summary_file is not None:
        args.summary_file.write_text(output + "\n\n" + verdict + "\n", encoding="utf-8")
    return code


if __name__ == "__main__":
    raise SystemExit(main())

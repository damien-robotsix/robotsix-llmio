"""Offline unit tests for the DeepSeek price-ceiling drift guard script.

The script itself is run on a schedule (not in CI), but its assertion logic
is pure and unit-testable — no network. Fixture endpoints use per-1M-token
prices (matching the ceiling comments); the helper converts them to the
per-token strings OpenRouter actually returns.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType
from typing import Any

import httpx
import pytest

# The price-ceiling check module loads ``scripts/check-price-ceilings.py``,
# which imports the OpenRouter DeepSeek transport layer (``_deepseek_model`` →
# ``model.py`` → ``from openai import AsyncStream``). The tests themselves are
# offline (mocked httpx), but that import chain makes the module uncollectable
# when the optional ``openai>=3`` extra is absent or stale — it would ERROR at
# fixture time. Skip the whole module *visibly* (N skipped with a reason)
# instead, so the gate shows WHY these tests are absent and auditing stays
# possible.
pytest.importorskip(
    "openai", minversion="3", reason="OpenRouter transport requires openai>=3"
)

_SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "check-price-ceilings.py"
_PER_MILLION = 1_000_000


@pytest.fixture(scope="module")
def check_module() -> ModuleType:
    """Load the drift-guard script once per test module."""
    spec = importlib.util.spec_from_file_location("check_price_ceilings", _SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    # Register under the spec name so dataclass annotations (and anything else
    # that looks itself up via ``sys.modules[__name__]``) resolve.
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _endpoint(
    provider: str,
    *,
    prompt: float,
    completion: float,
    cache_read: float = 0.0,
    status: int | None = 0,
) -> dict[str, Any]:
    """Build an endpoint dict with per-1M prices expressed as per-token strings."""
    return {
        "provider_name": provider,
        "status": status,
        "pricing": {
            "prompt": str(prompt / _PER_MILLION),
            "completion": str(completion / _PER_MILLION),
            "input_cache_read": str(cache_read / _PER_MILLION),
        },
    }


def _tier_check(check_module: ModuleType) -> Any:
    return check_module.TierCheck(
        label="test",
        model="deepseek/deepseek-v4-pro",
        ceiling={"prompt": 0.90, "completion": 2.40},
        ignore=("DigitalOcean",),
    )


# --- _price_per_million (pure) ------------------------------------------------


def test_price_per_million_converts_per_token_to_per_million(check_module):
    assert check_module._price_per_million("0.00000066") == pytest.approx(0.66)
    assert check_module._price_per_million("0.0000024") == pytest.approx(2.40)
    assert check_module._price_per_million(None) == 0.0


# --- check_tier (pure) --------------------------------------------------------


def test_check_tier_healthy_when_deepseek_and_three_others_admitted(check_module):
    endpoints = [
        _endpoint("DeepSeek", prompt=0.66, completion=1.98, cache_read=0.022),
        _endpoint("StreamLake", prompt=0.792, completion=2.376, cache_read=0.03),
        _endpoint("Baidu", prompt=0.80, completion=2.00, cache_read=0.04),
        _endpoint("Tail", prompt=1.13, completion=2.26, cache_read=0.02),
    ]
    report = check_module.check_tier(_tier_check(check_module), endpoints)

    assert report.failures == []
    assert report.preferred_admitted is True
    assert report.reference_cache_read == pytest.approx(0.022)
    # The $1.13 tail is over the ceiling, so only 3 are admitted.
    assert report.admitted_indices == {0, 1, 2}


def test_check_tier_flags_preferred_provider_above_ceiling(check_module):
    endpoints = [
        _endpoint("DeepSeek", prompt=1.20, completion=1.98),
        _endpoint("StreamLake", prompt=0.792, completion=2.376),
        _endpoint("Baidu", prompt=0.80, completion=2.00),
        _endpoint("GMICloud", prompt=0.85, completion=2.10),
    ]
    report = check_module.check_tier(_tier_check(check_module), endpoints)

    assert report.preferred_admitted is False
    assert any("preferred provider" in failure for failure in report.failures)


def test_check_tier_flags_insufficient_healthy_endpoints(check_module):
    endpoints = [
        _endpoint("DeepSeek", prompt=0.66, completion=1.98, status=0),
        _endpoint("StreamLake", prompt=0.792, completion=2.376, status=-2),
        _endpoint("Baidu", prompt=0.80, completion=2.00, status=-5),
    ]
    report = check_module.check_tier(_tier_check(check_module), endpoints)

    assert any("only 1 healthy endpoint" in failure for failure in report.failures)


def test_check_tier_flags_cache_read_outlier(check_module):
    endpoints = [
        _endpoint("DeepSeek", prompt=0.66, completion=1.98, cache_read=0.022),
        _endpoint("StreamLake", prompt=0.792, completion=2.376, cache_read=0.03),
        _endpoint("Baidu", prompt=0.80, completion=2.00, cache_read=0.04),
        _endpoint("Sneaky", prompt=0.70, completion=2.00, cache_read=0.20),
    ]
    report = check_module.check_tier(_tier_check(check_module), endpoints)

    assert any("cache-read" in failure for failure in report.failures)


def test_check_tier_ignored_provider_not_admitted(check_module):
    endpoints = [
        _endpoint("DeepSeek", prompt=0.66, completion=1.98),
        _endpoint("StreamLake", prompt=0.792, completion=2.376),
        _endpoint("Baidu", prompt=0.80, completion=2.00),
        _endpoint("DigitalOcean", prompt=0.87, completion=1.74, cache_read=0.174),
    ]
    report = check_module.check_tier(_tier_check(check_module), endpoints)

    # DigitalOcean is within the ceiling but ignored, so it must not count as
    # admitted (and its 7.9x cache-read must not trip the cache-read check).
    assert report.admitted_indices == {0, 1, 2}
    assert report.failures == []


def test_check_tier_flags_missing_preferred_provider_as_warning(check_module):
    endpoints = [
        _endpoint("StreamLake", prompt=0.792, completion=2.376),
        _endpoint("Baidu", prompt=0.80, completion=2.00),
        _endpoint("GMICloud", prompt=0.85, completion=2.10),
    ]
    report = check_module.check_tier(_tier_check(check_module), endpoints)

    assert report.failures == []
    assert any("has no endpoints" in warning for warning in report.warnings)


def test_cheap_tier_prefers_deepinfra_and_guard_passes(check_module):
    """The baked cheap tier must not prefer a provider that breaks its own
    ceiling. DeepSeek repriced flash to ~$0.44/$1.32 (2026-08-31), above the
    $0.10/$0.20 cheap ceiling, so ``FALLBACK_LEVEL1`` prefers a stable cheap
    upstream (DeepInfra) — otherwise rule 1 of the guard fails on the repriced
    DeepSeek endpoint."""
    cheap = next(t for t in check_module._TIERS if t.label == "cheap (level 1)")
    assert cheap.preferred_provider == "DeepInfra"

    report = check_module.check_tier(
        cheap,
        [
            _endpoint("DeepInfra", prompt=0.08, completion=0.18, cache_read=0.02),
            _endpoint("OpenInference", prompt=0.03, completion=0.16, cache_read=0.01),
            _endpoint("Makora", prompt=0.09, completion=0.195, cache_read=0.02),
            # DeepSeek's own repriced endpoint is over the ceiling — not admitted,
            # and (being non-preferred now) no longer a rule-1 failure.
            _endpoint("DeepSeek", prompt=0.44, completion=1.32, cache_read=0.007),
        ],
    )

    assert report.failures == []
    assert report.preferred_admitted is True
    # DeepSeek (index 3) is over the ceiling, so only the three cheap ones admit.
    assert report.admitted_indices == {0, 1, 2}


def test_cheap_tier_bars_wafer_cache_read_outlier(check_module):
    """Regression for the 2026-10-12 drift failure: Wafer lists inside the
    cheap ceiling ($0.130/$0.230) but its cache-read ($0.120/1M) is ~8x
    DeepInfra's ($0.015/1M), a cost ``max_price`` cannot see. The baked cheap
    tier now carries ``ignore_providers=["Wafer"]``, so the guard must not
    admit it and must not flag a cache-read drift."""
    cheap = next(t for t in check_module._TIERS if t.label == "cheap (level 1)")
    assert "Wafer" in cheap.ignore

    report = check_module.check_tier(
        cheap,
        [
            _endpoint("DeepInfra", prompt=0.060, completion=0.180, cache_read=0.015),
            _endpoint("DigitalOcean", prompt=0.119, completion=0.238, cache_read=0.024),
            _endpoint("Wafer", prompt=0.130, completion=0.230, cache_read=0.120),
            _endpoint("BaseTen", prompt=0.130, completion=0.260, cache_read=0.028),
            _endpoint("Venice", prompt=0.130, completion=0.260, cache_read=0.028),
        ],
    )

    assert report.failures == []
    assert report.preferred_admitted is True
    # Wafer (index 2) is ignored, so it must not be admitted.
    assert report.admitted_indices == {0, 1, 3, 4}


# --- fetch_endpoints (mocked httpx) -------------------------------------------


def test_healthy_endpoints_with_real_status_shapes_no_false_failure(
    check_module,
):
    """Regression: status ``0`` (healthy) must NOT produce an empty healthy list.

    OpenRouter's endpoint status is an integer (0 = healthy, negative =
    degraded/disabled).  An earlier bug mapped ``0`` to ``""``, causing every
    check to report zero healthy endpoints regardless of reality.
    """
    endpoints = [
        _endpoint("DeepSeek", prompt=0.66, completion=1.98, status=0),
        _endpoint("StreamLake", prompt=0.792, completion=2.376, status=-2),
        _endpoint("Baidu", prompt=0.80, completion=2.00, status=None),
        _endpoint("GMICloud", prompt=0.85, completion=2.10, status=0),
    ]
    # Baidu has status=None (absent key) → counted as healthy.
    report = check_module.check_tier(_tier_check(check_module), endpoints)

    assert report.failures == []
    assert report.admitted_indices == {0, 1, 2, 3}


def test_fetch_endpoints_hits_model_endpoints_path(check_module):
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/api/v1/models/deepseek/deepseek-v4-pro/endpoints"
        return httpx.Response(
            200,
            json={"data": {"endpoints": [{"provider_name": "DeepSeek"}]}},
        )

    client = httpx.Client(transport=httpx.MockTransport(handler))
    result = check_module.fetch_endpoints(client, "deepseek/deepseek-v4-pro")

    assert result == [{"provider_name": "DeepSeek"}]


# --- --fix: source rewriting ---------------------------------------------------

_TIER_PY = """\
FALLBACK_LEVEL1 = TierLevelConfig(
    model="openrouter-deepseek/deepseek-v4-flash-20260731",
    max_tokens=16384,
    provider_kwargs={
        "preferred_provider": "DeepInfra",
        "ignore_providers": ["Wafer"],
    },
)

FALLBACK_LEVEL2 = TierLevelConfig(
    model="openrouter-deepseek/deepseek-v4-flash-20260731",
    max_tokens=65536,
    provider_kwargs={
        "preferred_provider": "DeepInfra",
        "max_price_prompt": 0.13,
        "max_price_completion": 0.26,
    },
)

FALLBACK_LEVEL3 = TierLevelConfig(
    model="openrouter-deepseek/deepseek-v4-pro-0813",
    max_tokens=131072,
    provider_kwargs={
        "preferred_provider": "StreamLake",
        "max_price_prompt": 1.16,
        "max_price_completion": 3.40,
    },
)
"""

_DEEPSEEK_PY = """\
DEFAULT_MAX_PRICE_CHEAP: dict[str, float] = {"prompt": 0.13, "completion": 0.26}

#: Upstream providers barred from capable-tier fallback routing.
DEFAULT_IGNORE_CAPABLE: tuple[str, ...] = (
    "DigitalOcean",
    "CoreWeave",
    "Ionstream",
    "Alibaba",
)


def build_provider_routing() -> None:
    pass
"""


def _fixed_source(check_module, tmp_path):
    root = tmp_path / "repo"
    (root / "src/robotsix_llmio/config").mkdir(parents=True)
    (root / "src/robotsix_llmio/openrouter").mkdir(parents=True)
    (root / check_module._TIER_SOURCE).write_text(_TIER_PY)
    (root / check_module._DEEPSEEK_SOURCE).write_text(_DEEPSEEK_PY)
    return root


def test_extend_ignore_list_in_block_appends_to_existing_list(check_module):
    out = check_module.extend_ignore_list_in_block(
        _TIER_PY,
        "FALLBACK_LEVEL1",
        [("Sail Research", 0.1)],
        preferred_provider="DeepInfra",
        reference_cache_read=0.015,
        today="2026-10-13",
    )
    block = out.split("FALLBACK_LEVEL2")[0]
    assert '"ignore_providers": [' in block
    assert '"Wafer",' in block and '"Sail Research",' in block
    assert "# 2026-10-13 drift-fix: cache-read $0.100/M vs DeepInfra $0.015/M" in block
    # Level 2 untouched, and the module still parses.
    assert out.split("FALLBACK_LEVEL2")[1].count("ignore_providers") == 0
    compile(out.replace("TierLevelConfig", "dict"), "tier.py", "exec")


def test_extend_ignore_list_in_block_inserts_missing_key(check_module):
    out = check_module.extend_ignore_list_in_block(
        _TIER_PY,
        "FALLBACK_LEVEL2",
        [("Wafer", 0.12)],
        preferred_provider="DeepInfra",
        reference_cache_read=0.015,
        today="2026-10-13",
    )
    block = out.split("FALLBACK_LEVEL2")[1].split("FALLBACK_LEVEL3")[0]
    assert '"preferred_provider": "DeepInfra",\n        "ignore_providers": [' in block
    assert '"Wafer",' in block
    compile(out.replace("TierLevelConfig", "dict"), "tier.py", "exec")


def test_extend_ignore_list_in_block_is_idempotent(check_module):
    kwargs = {
        "preferred_provider": "DeepInfra",
        "reference_cache_read": 0.015,
        "today": "d",
    }
    once = check_module.extend_ignore_list_in_block(
        _TIER_PY, "FALLBACK_LEVEL1", [("Wafer", 0.12)], **kwargs
    )
    assert once == _TIER_PY


def test_extend_ignore_list_in_block_rejects_unknown_block(check_module):
    with pytest.raises(ValueError, match="FALLBACK_LEVEL9"):
        check_module.extend_ignore_list_in_block(
            _TIER_PY,
            "FALLBACK_LEVEL9",
            [("X", 1.0)],
            preferred_provider="p",
            reference_cache_read=None,
            today="d",
        )


def test_extend_ignore_tuple_appends_with_audit_comment(check_module):
    out = check_module.extend_ignore_tuple(
        _DEEPSEEK_PY,
        "DEFAULT_IGNORE_CAPABLE",
        [("Sail Research", 0.1), ("Alibaba", 0.112)],
        preferred_provider="StreamLake",
        reference_cache_read=0.022,
        today="2026-10-13",
    )
    assert out.count("Alibaba") == 1  # already present → not duplicated
    assert (
        '    "Alibaba",\n    # 2026-10-13 drift-fix: cache-read '
        '$0.100/M vs StreamLake $0.022/M\n    "Sail Research",\n)\n' in out
    )
    namespace: dict[str, Any] = {}
    exec(out, namespace)  # noqa: S102 - verifying the rewritten literal parses
    assert namespace["DEFAULT_IGNORE_CAPABLE"][-1] == "Sail Research"


def test_apply_cache_read_fixes_edits_the_right_file_per_tier(check_module, tmp_path):
    root = _fixed_source(check_module, tmp_path)
    cheap = check_module.TierCheck(
        label="cheap (level 1)",
        model="m",
        ceiling={"prompt": 0.13, "completion": 0.26},
        ignore=("Wafer",),
        preferred_provider="DeepInfra",
        ignore_source=(check_module._TIER_SOURCE, "FALLBACK_LEVEL1"),
    )
    capable = check_module.TierCheck(
        label="capable (level 3)",
        model="m3",
        ceiling={"prompt": 1.16, "completion": 3.40},
        ignore=("DigitalOcean", "CoreWeave", "Ionstream", "Alibaba"),
        preferred_provider="StreamLake",
        ignore_source=(check_module._DEEPSEEK_SOURCE, "DEFAULT_IGNORE_CAPABLE"),
    )
    # Real shape from run 38369768361 (2026-10-13): Sail Research admitted at
    # $0.48/$1.98 with cache-read $0.100 vs StreamLake $0.022.
    capable_eps = [
        _endpoint("StreamLake", prompt=0.66, completion=1.98, cache_read=0.022),
        _endpoint("Novita", prompt=0.99, completion=2.97, cache_read=0.033),
        _endpoint("GMICloud", prompt=1.056, completion=3.168, cache_read=0.035),
        _endpoint("Sail Research", prompt=0.48, completion=1.98, cache_read=0.100),
        _endpoint("Sail Research", prompt=0.48, completion=1.98, cache_read=0.100),
    ]
    cheap_eps = [
        _endpoint("DeepInfra", prompt=0.06, completion=0.18, cache_read=0.015),
        _endpoint("DigitalOcean", prompt=0.119, completion=0.238, cache_read=0.024),
        _endpoint("BaseTen", prompt=0.13, completion=0.26, cache_read=0.028),
        _endpoint("Venice", prompt=0.13, completion=0.26, cache_read=0.028),
    ]
    reports = [
        check_module.check_tier(cheap, cheap_eps),
        check_module.check_tier(capable, capable_eps),
    ]
    assert reports[0].cache_read_outliers == []
    assert reports[1].cache_read_outliers == [("Sail Research", pytest.approx(0.1))]

    applied = check_module.apply_cache_read_fixes(
        reports, repo_root=root, today="2026-10-13"
    )

    assert applied == [
        "capable (level 3): barred Sail Research in "
        f"{check_module._DEEPSEEK_SOURCE} (DEFAULT_IGNORE_CAPABLE)"
    ]
    assert (root / check_module._TIER_SOURCE).read_text() == _TIER_PY
    deepseek = (root / check_module._DEEPSEEK_SOURCE).read_text()
    assert '"Sail Research",' in deepseek
    # The extended list makes the tier green again.
    extended = check_module.TierCheck(
        label=capable.label,
        model=capable.model,
        ceiling=capable.ceiling,
        ignore=(*capable.ignore, "Sail Research"),
        preferred_provider=capable.preferred_provider,
    )
    assert check_module.check_tier(extended, capable_eps).failures == []


def test_main_fix_writes_summary_and_exits_zero_when_only_rule3_failed(
    check_module, tmp_path, monkeypatch
):
    root = _fixed_source(check_module, tmp_path)
    cheap_eps = [
        _endpoint("DeepInfra", prompt=0.06, completion=0.18, cache_read=0.015),
        _endpoint("DigitalOcean", prompt=0.119, completion=0.238, cache_read=0.024),
        _endpoint("BaseTen", prompt=0.13, completion=0.26, cache_read=0.028),
        _endpoint("Venice", prompt=0.13, completion=0.26, cache_read=0.028),
    ]
    # A synthetic outlier: the real one (Sail Research, run 38369768361) was
    # barred by the first auto-fix PR, after which a fixture named after it is
    # no longer admitted and the "before" run is green — the test must not
    # depend on the library's live ignore lists.
    capable_eps = [
        _endpoint("StreamLake", prompt=0.66, completion=1.98, cache_read=0.022),
        _endpoint("Novita", prompt=0.99, completion=2.97, cache_read=0.033),
        _endpoint("GMICloud", prompt=1.056, completion=3.168, cache_read=0.035),
        _endpoint("Sneaky Cache", prompt=0.48, completion=1.98, cache_read=0.100),
    ]
    by_model = {
        check_module._TIERS[0].model: cheap_eps,
        check_module._TIERS[1].model: cheap_eps,
        check_module._TIERS[2].model: capable_eps,
    }
    monkeypatch.setattr(
        check_module,
        "fetch_endpoints",
        lambda client, model, base_url=None: by_model[model],
    )
    summary = tmp_path / "summary.md"

    # Without --fix the run is red on rule 3 ...
    assert check_module.main(["--repo-root", str(root)]) == 1
    # ... with --fix the capable tier is re-checked with Sail Research barred.
    code = check_module.main(
        ["--fix", "--repo-root", str(root), "--summary-file", str(summary)]
    )
    assert code == 0
    text = summary.read_text()
    assert "## Applied fixes (--fix)" in text
    assert "barred Sneaky Cache" in text
    assert "## After fix" in text
    assert '"Sneaky Cache",' in (root / check_module._DEEPSEEK_SOURCE).read_text()
    # Guard the premise: none of the fixture providers is in a live ignore list.
    assert not {"Sneaky Cache", "StreamLake", "Novita", "GMICloud"} & set(
        check_module._TIERS[2].ignore
    )


def test_main_fix_still_fails_on_rule1_and_rule2(check_module, tmp_path, monkeypatch):
    root = _fixed_source(check_module, tmp_path)
    # Preferred provider priced above the ceiling, only two healthy endpoints:
    # nothing --fix can do.
    eps = [
        _endpoint("DeepInfra", prompt=0.50, completion=1.00, cache_read=0.015),
        _endpoint("BaseTen", prompt=0.13, completion=0.26, cache_read=0.028),
        _endpoint("Venice", prompt=0.13, completion=0.26, cache_read=0.028),
    ]
    capable_eps = [
        _endpoint("StreamLake", prompt=0.66, completion=1.98, cache_read=0.022),
        _endpoint("Novita", prompt=0.99, completion=2.97, cache_read=0.033),
        _endpoint("GMICloud", prompt=1.056, completion=3.168, cache_read=0.035),
    ]
    by_model = {
        check_module._TIERS[0].model: eps,
        check_module._TIERS[1].model: eps,
        check_module._TIERS[2].model: capable_eps,
    }
    monkeypatch.setattr(
        check_module,
        "fetch_endpoints",
        lambda client, model, base_url=None: by_model[model],
    )
    assert check_module.main(["--fix", "--repo-root", str(root)]) == 1
    assert (root / check_module._TIER_SOURCE).read_text() == _TIER_PY

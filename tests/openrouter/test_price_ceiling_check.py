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
    status: str = "healthy",
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
        _endpoint("DeepSeek", prompt=0.66, completion=1.98, status="healthy"),
        _endpoint("StreamLake", prompt=0.792, completion=2.376, status="degraded"),
        _endpoint("Baidu", prompt=0.80, completion=2.00, status="unhealthy"),
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


# --- fetch_endpoints (mocked httpx) -------------------------------------------


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

"""Tests for the ``chain`` key in :func:`load_tier_config` and the persisted
config helpers (dump / save / load-with-migration)."""

from __future__ import annotations

import json

import pytest

from robotsix_llmio.config.loader import (
    TierConfigLoadError,
    dump_tier_config,
    load_tier_config,
    load_tier_config_file,
    save_tier_config_file,
)
from robotsix_llmio.config.tier import TierConfig, _legacy_chain

# --------------------------------------------------------------------------- #
#  ``chain`` key in load_tier_config                                          #
# --------------------------------------------------------------------------- #


def test_chain_override_merges_over_baked_position():
    cfg = load_tier_config(
        {"chain": [{}, {"level2": {"model": "claudeSDK-sonnet"}}, {}]}
    )
    baked = TierConfig()
    assert cfg.chain[1].level2.model == "claudeSDK-sonnet"
    assert cfg.chain[0] == baked.chain[0]
    assert cfg.chain[2] == baked.chain[2]


def test_chain_override_length_sets_chain_length():
    cfg = load_tier_config({"chain": [{}]})
    assert len(cfg.chain) == 1
    assert cfg.chain[0] == TierConfig().chain[0]


def test_chain_override_extra_position_must_be_complete():
    slot = TierConfig().chain[2].model_dump()
    cfg = load_tier_config({"chain": [{}, {}, {}, slot]})
    assert len(cfg.chain) == 4
    with pytest.raises(TierConfigLoadError):
        load_tier_config({"chain": [{}, {}, {}, {"level1": slot["level1"]}]})


def test_chain_mixed_with_legacy_keys_raises():
    with pytest.raises(TierConfigLoadError, match="mixes"):
        load_tier_config({"chain": [{}], "default": {}})


def test_chain_not_a_list_raises():
    with pytest.raises(TierConfigLoadError, match="list"):
        load_tier_config({"chain": {"level1": {}}})


def test_dump_round_trips_in_chain_shape():
    cfg = load_tier_config({"fallback": {"level1": {"model": "claudeSDK-sonnet"}}})
    dumped = dump_tier_config(cfg)
    assert "chain" in dumped
    assert "default" not in dumped and "fallback" not in dumped
    json.dumps(dumped)  # JSON-compatible
    assert load_tier_config(dumped) == cfg
    assert TierConfig.model_validate(dumped) == cfg


# --------------------------------------------------------------------------- #
#  Persisted config files                                                     #
# --------------------------------------------------------------------------- #


def test_missing_file_writes_fresh_chain_config(tmp_path):
    path = tmp_path / "sub" / "tiers.json"
    cfg = load_tier_config_file(path)
    assert cfg == TierConfig()
    on_disk = json.loads(path.read_text())
    assert set(on_disk) == {"chain", "failover", "vision"}
    assert len(on_disk["chain"]) == 3


def test_missing_file_read_only_does_not_write(tmp_path):
    path = tmp_path / "tiers.json"
    assert load_tier_config_file(path, migrate=False) == TierConfig()
    assert not path.exists()


def test_legacy_file_loads_and_is_repersisted_in_chain_shape(tmp_path):
    path = tmp_path / "tiers.json"
    legacy = {
        "default": {"level2": {"model": "claudeSDK-sonnet"}},
        "fallback": {"level1": {"model": "openrouter-some/model"}},
        "failover": {"window_seconds": 120},
    }
    path.write_text(json.dumps(legacy))

    cfg = load_tier_config_file(path)

    assert len(cfg.chain) == 2
    assert cfg.chain[0].level2.model == "claudeSDK-sonnet"
    assert cfg.chain[1].level1.model == "openrouter-some/model"
    assert cfg.chain[0].level1 == _legacy_chain()[0].level1
    assert cfg.failover.window_seconds == 120

    on_disk = json.loads(path.read_text())
    assert "default" not in on_disk and "fallback" not in on_disk
    assert len(on_disk["chain"]) == 2
    # Second startup reads the migrated file and gets the same config.
    assert load_tier_config_file(path) == cfg


def test_legacy_file_read_only_is_left_untouched(tmp_path):
    path = tmp_path / "tiers.json"
    text = json.dumps({"default": {}})
    path.write_text(text)
    cfg = load_tier_config_file(path, migrate=False)
    assert len(cfg.chain) == 2
    assert path.read_text() == text


def test_chain_file_is_not_rewritten(tmp_path):
    path = tmp_path / "tiers.json"
    text = json.dumps({"chain": [{}, {}]})
    path.write_text(text)
    cfg = load_tier_config_file(path)
    assert len(cfg.chain) == 2
    assert path.read_text() == text


def test_save_writes_chain_shape(tmp_path):
    path = tmp_path / "tiers.json"
    save_tier_config_file(TierConfig(), path)
    assert json.loads(path.read_text()) == dump_tier_config(TierConfig())
    assert [p.name for p in tmp_path.iterdir()] == ["tiers.json"]


@pytest.mark.parametrize("content", ["not json", "[1, 2]"])
def test_malformed_file_raises_load_error(tmp_path, content):
    path = tmp_path / "tiers.json"
    path.write_text(content)
    with pytest.raises(TierConfigLoadError):
        load_tier_config_file(path)


def _readonly_dir(monkeypatch):
    """Simulate a read-only config directory: every temp-file create fails."""

    def _deny(*_args, **_kwargs):
        raise PermissionError(13, "Read-only file system")

    monkeypatch.setattr("robotsix_llmio.config.loader.tempfile.mkstemp", _deny)


def test_legacy_file_in_readonly_dir_still_loads(tmp_path, monkeypatch, caplog):
    path = tmp_path / "tiers.json"
    text = json.dumps({"default": {"level2": {"model": "claudeSDK-sonnet"}}})
    path.write_text(text)
    _readonly_dir(monkeypatch)
    with caplog.at_level("WARNING", logger="robotsix_llmio.config.loader"):
        cfg = load_tier_config_file(path)
    assert len(cfg.chain) == 2
    assert cfg.chain[0].level2.model == "claudeSDK-sonnet"
    assert path.read_text() == text
    assert "Could not persist" in caplog.text


def test_missing_file_in_readonly_dir_yields_baked(tmp_path, monkeypatch):
    path = tmp_path / "tiers.json"
    _readonly_dir(monkeypatch)
    assert load_tier_config_file(path) == TierConfig()
    assert not path.exists()

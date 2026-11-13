## Checklist

See `AGENT.md` at the repo root for the repo's doc-sync rules behind the conditional items below.

- [ ] The PR title is a short, clear summary of the change.
- [ ] Unit tests are added or updated and all pass (`pytest`).
- [ ] Lint passes (`ruff check .`).
- [ ] Type-check passes (`mypy src/`).
- [ ] Documentation is updated if needed.
- [ ] If tests were split/renamed/removed: ARCHITECTURE.md 'Test coverage strategy' `tests/` tree updated.
- [ ] If a runtime env var was added: README 'Configuration' table row added (not `tests/.env.example`).
- [ ] If `config/tier.py` defaults changed: legacy `default`/`fallback` mapping checked in both `load_tier_config` and `TierConfig` (`_legacy_chain`), with a test covering both.
- [ ] If a CI job was added/renamed/removed: branch-protection required checks updated; old and new check contexts listed in this description.
- [ ] If a CVE ignore was added: both `[tool.uv.audit].ignore` in `pyproject.toml` and `pip-audit-ignore-vulns` in `.github/workflows/ci.yml` updated.
- [ ] The change is ready for review.

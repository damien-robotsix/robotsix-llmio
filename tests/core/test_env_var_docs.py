"""Env-var README / docs-reference drift check.

The package centralises every runtime env-var name as a module-level
``_ENV_*`` constant (AGENT.md "Configuration conventions") and documents the
full runtime env interface in two markdown tables — README's
``## Configuration`` and ``docs/config/reference.md``'s
``## Runtime environment variables``. This test reconciles the code against
both tables using only ``ast`` + ``pathlib`` (no package imports), so it runs
offline and without any optional extra:

- **Prefix guard** — every collected env constant must carry the ``_ENV_``
  prefix; a bare ``ENV_`` name (the historical ``ENV_LANGFUSE_*`` naming)
  fails.
- **Centralisation guard** — every ``os.environ`` / ``os.getenv`` read must
  use a constant, never a string literal.
- **Dead-constant guard** — every ``_ENV_*`` constant must be referenced by
  at least one ``os.environ`` read somewhere in the package.
- **Per-table sync** — each verified table's ``Variable`` column must equal
  the canonical set of referenced env-var values, bidirectionally.
- **Sibling sweep** — no other ``*.md`` file may document an env-var name in
  a table's first column; a future third copy fails CI until it is added to
  the verified set and reconciled.

The doc comparison is names only: the ``Purpose`` / ``Default`` / ``Type`` /
``Read by`` / ``Effect`` text is not validated.
"""

from __future__ import annotations

import ast
import re
from pathlib import Path

REPO_ROOT = Path(__file__).parents[2]
SRC_ROOT = REPO_ROOT / "src" / "robotsix_llmio"

_ENV_CONSTANT_RE = re.compile(r"^(ENV|_ENV)_[A-Z0-9_]+$")

# Verified doc tables, keyed by heading anchor. A future doc copy must be
# added here (and reconciled) before the sibling sweep stops failing on it.
VERIFIED_TABLES: dict[Path, str] = {
    REPO_ROOT / "README.md": "## Configuration",
    REPO_ROOT / "docs" / "config" / "reference.md": "## Runtime environment variables",
}

_EXCLUDED_MD = {"CHANGELOG.md", "docs/changelog.md"}
_GENERATED_DIRS = {"site", "build", ".venv", ".git"}


def _collect_env_usage() -> tuple[dict[str, str], set[str], list[str]]:
    """AST-scan the package for env constants and their ``os.environ`` reads.

    Returns (constants, referenced, literal_keys): *constants* maps each
    module-level env-constant name to its string value; *referenced* is the
    set of constant names used as an ``os.environ`` key anywhere in the
    package; *literal_keys* is a list of ``"path:line"`` strings for reads
    whose key is a string literal.
    """
    constants: dict[str, str] = {}
    referenced: set[str] = set()
    literal_keys: list[str] = []

    for path in sorted(SRC_ROOT.rglob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in tree.body:
            if not isinstance(node, ast.Assign):
                continue
            if len(node.targets) != 1 or not isinstance(node.targets[0], ast.Name):
                continue
            name = node.targets[0].id
            if not _ENV_CONSTANT_RE.match(name):
                continue
            if not isinstance(node.value, ast.Constant) or not isinstance(
                node.value.value, str
            ):
                continue
            constants[name] = node.value.value
        for node in ast.walk(tree):
            key = _environ_key(node)
            if key is None:
                continue
            if isinstance(key, ast.Constant) and isinstance(key.value, str):
                literal_keys.append(f"{path.relative_to(REPO_ROOT)}:{key.lineno}")
            elif isinstance(key, ast.Name):
                referenced.add(key.id)

    return constants, referenced, literal_keys


def _environ_key(node: ast.AST) -> ast.AST | None:
    """The env-var key expression for *node*, or None when it isn't a read.

    Recognises ``os.environ.get(key)``, ``os.getenv(key)`` and
    ``os.environ[key]``.
    """
    if isinstance(node, ast.Call):
        func = node.func
        if not isinstance(func, ast.Attribute) or func.attr not in {"get", "getenv"}:
            return None
        if not node.args:
            return None
        owner = func.value
        if (
            isinstance(owner, ast.Attribute)
            and owner.attr == "environ"
            and isinstance(owner.value, ast.Name)
            and owner.value.id == "os"
        ):
            return node.args[0]
        if isinstance(owner, ast.Name) and owner.id == "os":
            return node.args[0]
    if isinstance(node, ast.Subscript):
        value = node.value
        if (
            isinstance(value, ast.Attribute)
            and value.attr == "environ"
            and isinstance(value.value, ast.Name)
            and value.value.id == "os"
        ):
            return node.slice
    return None


def _table_variable_tokens(path: Path, heading: str) -> set[str]:
    """The backtick-quoted names in the first table column under *heading*."""
    lines = path.read_text(encoding="utf-8").splitlines()
    try:
        header_index = lines.index(heading)
    except ValueError:
        raise AssertionError(f"{path}: heading {heading!r} not found") from None
    for index in range(header_index + 1, len(lines)):
        if not lines[index].lstrip().startswith("|"):
            continue
        tokens: set[str] = set()
        row = index + 2
        while row < len(lines) and lines[row].lstrip().startswith("|"):
            columns = lines[row].split("|")
            if len(columns) >= 3:
                for match in re.finditer(r"`([^`]+)`", columns[1]):
                    for part in match.group(1).split("/"):
                        part = part.strip()
                        if part:
                            tokens.add(part)
            row += 1
        return tokens
    raise AssertionError(f"{path}: no table under heading {heading!r}")


def _sibling_doc_offenders(canonical: set[str]) -> list[str]:
    """Markdown tables outside the verified set that document a canonical name."""
    offenders: list[str] = []
    for path in REPO_ROOT.rglob("*.md"):
        rel = path.relative_to(REPO_ROOT).as_posix()
        if rel in _EXCLUDED_MD or path in VERIFIED_TABLES:
            continue
        if any(part in _GENERATED_DIRS or part.startswith(".") for part in path.parts):
            continue
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.lstrip().startswith("|"):
                continue
            columns = line.split("|")
            if len(columns) < 3:
                continue
            for match in re.finditer(r"`([^`]+)`", columns[1]):
                for part in match.group(1).split("/"):
                    if part.strip() in canonical:
                        offenders.append(f"{rel} ({part.strip()})")
    return sorted(set(offenders))


def _canonical_env_vars(constants: dict[str, str], referenced: set[str]) -> set[str]:
    """The values of the env constants actually referenced by an env read."""
    return {constants[name] for name in referenced if name in constants}


def test_env_constants_are_prefixed_and_used() -> None:
    constants, referenced, literal_keys = _collect_env_usage()

    prefix_violations = sorted(n for n in constants if not n.startswith("_ENV_"))
    assert prefix_violations == [], (
        f"env constants missing _ENV_ prefix: {prefix_violations}"
    )

    assert literal_keys == [], (
        f"os.environ reads using a literal key instead of a constant: {literal_keys}"
    )

    dead_constants = sorted(n for n in constants if n not in referenced)
    assert dead_constants == [], (
        f"_ENV_* constants never read via os.environ: {dead_constants}"
    )


def test_doc_tables_match_code() -> None:
    constants, referenced, _ = _collect_env_usage()
    canonical = _canonical_env_vars(constants, referenced)

    for path, heading in VERIFIED_TABLES.items():
        documented = _table_variable_tokens(path, heading)
        missing = sorted(canonical - documented)
        assert missing == [], f"{path} missing doc rows for: {missing}"
        undocumented = sorted(documented - canonical)
        assert undocumented == [], f"{path} documents unknown env vars: {undocumented}"


def test_no_other_markdown_documents_env_vars() -> None:
    constants, referenced, _ = _collect_env_usage()
    canonical = _canonical_env_vars(constants, referenced)

    offenders = _sibling_doc_offenders(canonical)
    assert offenders == [], (
        f"env vars documented outside the verified tables: {offenders}"
    )

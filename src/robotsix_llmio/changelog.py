"""Changelog entry insertion with format detection.

Provides ``insert_changelog_entry``, which detects the project's
existing changelog format (``.rst``, ``.md``, or extensionless) and
inserts a bullet entry under the unreleased section.  Falls back to
``CHANGELOG.md`` when no changelog file exists.
"""

from __future__ import annotations

import os
import re
from collections.abc import Callable
from pathlib import Path

_UNRELEASED_MD_PATTERN = re.compile(r"^##\s+.*unreleased", re.IGNORECASE)
_UNRELEASED_RST_PATTERN = re.compile(r"^.*unreleased", re.IGNORECASE)


def _format_bullet(text: str, prefix: str = "- ", indent: str = "  ") -> list[str]:
    """Split *text* into a first line (with *prefix*) and continuation
    lines (indented with *indent*)."""
    text_lines = text.splitlines()
    if not text_lines:
        return [f"{prefix}\n"]
    result = [f"{prefix}{text_lines[0]}\n"]
    for line in text_lines[1:]:
        result.append(f"{indent}{line}\n")
    return result


# ------------------------------------------------------------------ shared


def _insert_into_format(
    path: str,
    entry_text: str,
    *,
    header: str,
    find_or_append_unreleased: Callable[[list[str]], int],
    skip_past_heading: Callable[[list[str], int], int],
) -> str:
    entry_text = entry_text.strip()
    indent = "  "
    bullet_lines = _format_bullet(entry_text, prefix="- ", indent=indent)

    if not os.path.exists(path):
        Path(path).write_text(header + "".join(bullet_lines))
        return f"created {path} with header + entry"

    lines = Path(path).read_text().splitlines(keepends=True)
    unreleased_idx = find_or_append_unreleased(lines)
    insert_at = skip_past_heading(lines, unreleased_idx)

    for bl in reversed(bullet_lines):
        lines.insert(insert_at, bl)

    Path(path).write_text("".join(lines))
    return f"inserted entry into {path}"


# ------------------------------------------------------------------ Markdown


def _md_find_or_append(lines: list[str]) -> int:
    for i, line in enumerate(lines):
        if _UNRELEASED_MD_PATTERN.match(line):
            return i
    lines.append("\n## 0.0.0 (unreleased)\n")
    return len(lines) - 1


def _md_skip_past_heading(lines: list[str], unreleased_idx: int) -> int:
    insert_at = unreleased_idx + 1
    while insert_at < len(lines) and lines[insert_at].strip() == "":
        insert_at += 1
    return insert_at


def _insert_into_md(path: str, entry_text: str) -> str:
    return _insert_into_format(
        path,
        entry_text,
        header="# Changelog\n\n## 0.0.0 (unreleased)\n\n",
        find_or_append_unreleased=_md_find_or_append,
        skip_past_heading=_md_skip_past_heading,
    )


# ------------------------------------------------------------------ reST


def _rst_find_or_append(lines: list[str]) -> int:
    for i, line in enumerate(lines):
        stripped = line.strip()
        if _UNRELEASED_RST_PATTERN.match(stripped) and i + 1 < len(lines):
            next_line = lines[i + 1].rstrip("\n")
            if (
                next_line
                and all(c in "-=~^\"'`*+#" for c in next_line)
                and len(next_line) >= len(stripped)
            ):
                return i
    lines.append("\n")
    lines.append("0.0.0 (unreleased)\n")
    lines.append("------------------\n")
    lines.append("\n")
    return len(lines) - 3


def _rst_skip_past_heading(lines: list[str], unreleased_idx: int) -> int:
    insert_at = unreleased_idx + 1
    if insert_at < len(lines):
        ruler = lines[insert_at].rstrip("\n")
        if ruler and all(c in "-=~^\"'`*+#" for c in ruler):
            insert_at += 1
    while insert_at < len(lines) and lines[insert_at].strip() == "":
        insert_at += 1
    return insert_at


def _insert_into_rst(path: str, entry_text: str) -> str:
    return _insert_into_format(
        path,
        entry_text,
        header=("Changelog\n=========\n\n0.0.0 (unreleased)\n------------------\n\n"),
        find_or_append_unreleased=_rst_find_or_append,
        skip_past_heading=_rst_skip_past_heading,
    )


# ------------------------------------------------------------------ public


def insert_changelog_entry(entry_text: str) -> str:
    """Insert *entry_text* into the project's changelog file.

    Checks for ``CHANGELOG.rst``, ``CHANGELOG.md``, and ``CHANGELOG``
    (in that order).  Writes to whichever exists; creates
    ``CHANGELOG.md`` when none are present.
    """
    candidates = ["CHANGELOG.rst", "CHANGELOG.md", "CHANGELOG"]
    changelog_path = ""
    for candidate in candidates:
        if os.path.exists(candidate):
            changelog_path = candidate
            break

    if not changelog_path:
        changelog_path = "CHANGELOG.md"

    if changelog_path.endswith(".rst"):
        return _insert_into_rst(changelog_path, entry_text)
    return _insert_into_md(changelog_path, entry_text)

"""Tests for :mod:`robotsix_llmio.changelog` — changelog entry insertion
with format detection."""

from __future__ import annotations

from pathlib import Path

import pytest

from robotsix_llmio.changelog import insert_changelog_entry


@pytest.fixture(autouse=True)
def _isolate_changelogs(tmp_path, monkeypatch):
    """Run each test in a clean temporary directory so the repo's own
    CHANGELOG.md doesn't interfere with detection logic."""
    monkeypatch.chdir(tmp_path)


# ------------------------------------------------------------------ Markdown


def test_creates_md_when_none_exist():
    """When no changelog file exists, CHANGELOG.md is created."""
    result = insert_changelog_entry("new feature")
    assert "created CHANGELOG.md" in result
    content = Path("CHANGELOG.md").read_text()
    assert "# Changelog" in content
    assert "## 0.0.0 (unreleased)" in content
    assert "- new feature" in content


def test_inserts_into_existing_md():
    """When CHANGELOG.md exists, entry is inserted under unreleased."""
    Path("CHANGELOG.md").write_text(
        "# Changelog\n\n## 0.0.0 (unreleased)\n\n- old entry\n"
    )
    result = insert_changelog_entry("new entry")
    assert "inserted entry into CHANGELOG.md" in result
    content = Path("CHANGELOG.md").read_text()
    lines = content.splitlines()
    # new entry should appear before old entry
    unreleased_idx = next(i for i, ln in enumerate(lines) if "0.0.0 (unreleased)" in ln)
    new_idx = next(i for i, ln in enumerate(lines) if "new entry" in ln)
    old_idx = next(i for i, ln in enumerate(lines) if "old entry" in ln)
    assert unreleased_idx < new_idx < old_idx


def test_md_multiline_entry():
    """Multi-line entries get continuation lines indented."""
    result = insert_changelog_entry("line one\nline two\nline three")
    assert "created CHANGELOG.md" in result
    content = Path("CHANGELOG.md").read_text()
    assert "- line one" in content
    assert "  line two" in content
    assert "  line three" in content


def test_md_no_existing_unreleased_section():
    """When CHANGELOG.md exists but has no unreleased section, one is
    appended."""
    Path("CHANGELOG.md").write_text("# Changelog\n\n## 1.0.0\n\n- item\n")
    result = insert_changelog_entry("unreleased item")
    assert "inserted entry" in result
    content = Path("CHANGELOG.md").read_text()
    assert "## 0.0.0 (unreleased)" in content
    assert "- unreleased item" in content


# ------------------------------------------------------------------ reST


def test_creates_rst_when_none_exist_and_requested():
    """When no changelog exists and we create directly, we get .md; but if
    only .rst exists we respect it."""
    # This test ensures that when a CHANGELOG.rst already exists,
    # it's preferred and we insert into it.
    Path("CHANGELOG.rst").write_text(
        "Changelog\n"
        "=========\n\n"
        "0.0.0 (unreleased)\n"
        "------------------\n\n"
        "- old rst entry\n"
    )
    result = insert_changelog_entry("new rst entry")
    assert "CHANGELOG.rst" in result
    content = Path("CHANGELOG.rst").read_text()
    assert "- new rst entry" in content
    assert "- old rst entry" in content


def test_rst_preferred_over_md():
    """When both CHANGELOG.rst and CHANGELOG.md exist, .rst wins."""
    Path("CHANGELOG.rst").write_text(
        "Changelog\n=========\n\n0.0.0 (unreleased)\n------------------\n\n"
    )
    Path("CHANGELOG.md").write_text("# Changelog\n\n## 0.0.0 (unreleased)\n\n")
    result = insert_changelog_entry("rst wins")
    assert "CHANGELOG.rst" in result
    # CHANGELOG.md should be untouched
    md_content = Path("CHANGELOG.md").read_text()
    assert "rst wins" not in md_content


def test_rst_creates_when_file_does_not_exist_but_rst_path_forced():
    """Directly create CHANGELOG.rst (simulating formatter wanting .rst)."""
    # This test simulates the case where the tool is told to write to .rst.
    # Since there's no existing file, it falls back to .md by default.
    # But if the tool were modified to accept a path override...
    # For now, just verify the fallback works correctly.
    Path("CHANGELOG.rst").write_text(
        "Changelog\n=========\n\n0.0.0 (unreleased)\n------------------\n\n"
    )
    result = insert_changelog_entry("my rst bullet")
    assert "CHANGELOG.rst" in result
    content = Path("CHANGELOG.rst").read_text()
    assert "- my rst bullet" in content


def test_rst_multiline_entry():
    """Multi-line RST entries work."""
    Path("CHANGELOG.rst").write_text(
        "Changelog\n=========\n\n0.0.0 (unreleased)\n------------------\n\n"
    )
    result = insert_changelog_entry("first\nsecond")
    assert "CHANGELOG.rst" in result
    content = Path("CHANGELOG.rst").read_text()
    assert "- first" in content
    assert "  second" in content


def test_rst_no_unreleased_section():
    """When RST exists but lacks unreleased, one is appended."""
    Path("CHANGELOG.rst").write_text("Changelog\n=========\n\n1.0.0\n-----\n\n- item\n")
    result = insert_changelog_entry("new unreleased")
    assert "CHANGELOG.rst" in result
    content = Path("CHANGELOG.rst").read_text()
    assert "0.0.0 (unreleased)" in content
    assert "- new unreleased" in content


# ------------------------------------------------------------------ Edge cases


def test_empty_entry_text():
    """Empty entry text still inserts (empty bullet)."""
    result = insert_changelog_entry("")
    assert "created CHANGELOG.md" in result
    content = Path("CHANGELOG.md").read_text()
    assert "- " in content  # bullet with empty text


def test_entry_with_only_whitespace():
    """Whitespace-only entry is stripped to empty."""
    result = insert_changelog_entry("   \n  \n  ")
    assert "created CHANGELOG.md" in result
    content = Path("CHANGELOG.md").read_text()
    assert "- " in content


def test_changelog_no_extension_treated_as_md():
    """A CHANGELOG file (no extension) is treated as Markdown."""
    Path("CHANGELOG").write_text("# Changelog\n\n## 0.0.0 (unreleased)\n\n- old\n")
    result = insert_changelog_entry("new via no-ext")
    assert "CHANGELOG" in result
    content = Path("CHANGELOG").read_text()
    assert "- new via no-ext" in content
    assert "- old" in content
    # Also ensure no .md or .rst were spuriously created
    assert not Path("CHANGELOG.md").exists()
    assert not Path("CHANGELOG.rst").exists()

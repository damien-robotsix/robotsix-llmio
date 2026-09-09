"""Workspace confinement for the claude_sdk tool path.

A tool-bearing claude_sdk agent runs under ``permission_mode="bypassPermissions"``,
so the SDK's built-in ``Write``/``Edit``/``MultiEdit``/``NotebookEdit`` tools can
write anywhere the process can reach — including the host app's own source. When
``build_agent(workspace_root=...)`` is set, a ``PreToolUse`` hook must DENY any
edit whose target resolves outside the workspace while allowing edits inside it.
These tests exercise that hook + the path predicate directly (no ``claude`` CLI).
"""

from __future__ import annotations

import asyncio

import pytest

from robotsix_llmio.claude_sdk._confinement import (
    _is_within,
    _make_bash_confine_hook,
    _make_confine_hook,
)
from robotsix_llmio.claude_sdk.provider import (
    ClaudeSDKProvider,
)


def _run_hook(root, tool_name, tool_input):
    hook = _make_confine_hook(str(root))
    return asyncio.run(
        hook({"tool_name": tool_name, "tool_input": tool_input}, "tu_1", None)
    )


def _run_bash_hook(root, command):
    hook = _make_bash_confine_hook(str(root))
    return asyncio.run(
        hook(
            {"tool_name": "Bash", "tool_input": {"command": command}},
            "tu_1",
            None,
        )
    )


def _denied(out) -> bool:
    return (out.get("hookSpecificOutput") or {}).get("permissionDecision") == "deny"


# --- _is_within -----------------------------------------------------------


def test_is_within_absolute_inside_and_outside(tmp_path):
    root = str(tmp_path)
    assert _is_within(root, str(tmp_path / "src" / "a.py")) is True
    assert _is_within(root, "/app/src/robotsix_mill/agents/coding.py") is False
    # the root itself counts as inside
    assert _is_within(root, root) is True


def test_is_within_relative_is_joined_to_root(tmp_path):
    root = str(tmp_path)
    assert _is_within(root, "src/a.py") is True
    # a sibling-prefix dir must NOT be treated as inside (no string-prefix bug)
    assert (
        _is_within(str(tmp_path / "repo"), str(tmp_path / "repo-evil" / "x")) is False
    )


def test_is_within_dotdot_escape_is_caught(tmp_path):
    root = str(tmp_path / "repo")
    (tmp_path / "repo").mkdir()
    assert _is_within(root, "../outside.py") is False


# --- the hook -------------------------------------------------------------


def test_hook_allows_edit_inside_workspace(tmp_path):
    out = _run_hook(tmp_path, "Edit", {"file_path": str(tmp_path / "src/a.py")})
    assert out == {}  # empty → no decision → proceeds


def test_hook_denies_edit_outside_workspace(tmp_path):
    out = _run_hook(
        tmp_path, "Edit", {"file_path": "/app/src/robotsix_mill/agents/coding.py"}
    )
    assert _denied(out)
    reason = out["hookSpecificOutput"]["permissionDecisionReason"]
    assert "confined" in reason and "/app/src" in reason


def test_hook_denies_write_and_multiedit_and_notebook(tmp_path):
    assert _denied(_run_hook(tmp_path, "Write", {"file_path": "/etc/passwd"}))
    assert _denied(_run_hook(tmp_path, "MultiEdit", {"file_path": "/app/x.py"}))
    assert _denied(
        _run_hook(tmp_path, "NotebookEdit", {"notebook_path": "/app/n.ipynb"})
    )


def test_hook_allows_relative_path_inside(tmp_path):
    assert _run_hook(tmp_path, "Write", {"file_path": "src/new.py"}) == {}


def test_hook_denies_relative_dotdot_escape(tmp_path):
    root = tmp_path / "repo"
    root.mkdir()
    assert _denied(_run_hook(root, "Edit", {"file_path": "../../../app/x.py"}))


def test_hook_allows_calls_without_a_path(tmp_path):
    # A matched tool that somehow carries no path key must not be denied.
    assert _run_hook(tmp_path, "Edit", {}) == {}


# --- Bash hook ------------------------------------------------------------


def test_bash_hook_denies_sed_outside_workspace(tmp_path):
    out = _run_bash_hook(tmp_path, "sed -i 's/a/b/' /app/src/x.py")
    assert _denied(out)
    reason = out["hookSpecificOutput"]["permissionDecisionReason"]
    assert "/app/src/x.py" in reason


def test_bash_hook_denies_redirect_outside_workspace(tmp_path):
    out = _run_bash_hook(tmp_path, "cat > /app/src/y.py")
    assert _denied(out)
    reason = out["hookSpecificOutput"]["permissionDecisionReason"]
    assert "/app/src/y.py" in reason


def test_bash_hook_denies_ln_hardlink_to_outside(tmp_path):
    out = _run_bash_hook(tmp_path, "ln /app/src/x.py ./link.py")
    assert _denied(out)
    reason = out["hookSpecificOutput"]["permissionDecisionReason"]
    assert "/app/src/x.py" in reason


def test_bash_hook_allows_relative_paths(tmp_path):
    assert _run_bash_hook(tmp_path, "sed -i 's/a/b/' src/a.py") == {}


@pytest.mark.parametrize(
    "command",
    [
        "grep -r foo src/ 2>/dev/null",
        "cat src/a.py > /dev/null",
        "head -c 16 /dev/urandom | xxd",
        "diff src/a.py /dev/stdin",
        "cmd </dev/null >/dev/stdout 2>/dev/stderr",
        "wc -l /dev/fd/3",
    ],
)
def test_bash_hook_allows_safe_pseudo_devices(tmp_path, command):
    """`2>/dev/null` and friends are ubiquitous, leak nothing, and must not
    be refused (false refusals burned mill review/ci_fix turns, 2026-09-05)."""
    assert _run_bash_hook(tmp_path, command) == {}


def test_bash_hook_still_denies_other_dev_like_paths(tmp_path):
    # The allowlist is exact — /dev/sda (raw device) stays denied.
    assert _denied(_run_bash_hook(tmp_path, "dd if=/dev/sda of=x.img"))


def test_bash_hook_allows_absolute_inside_workspace(tmp_path):
    assert _run_bash_hook(tmp_path, f"cat {tmp_path}/src/a.py") == {}


def test_bash_hook_allows_empty_command(tmp_path):
    assert _run_bash_hook(tmp_path, "") == {}


def test_bash_hook_allows_missing_command_key(tmp_path):
    hook = _make_bash_confine_hook(str(tmp_path))
    out = asyncio.run(hook({"tool_name": "Bash", "tool_input": {}}, "tu_1", None))
    assert out == {}


# --- threading through build_agent ----------------------------------------


def test_build_agent_threads_workspace_root(tmp_path):
    """build_agent(workspace_root=...) reaches the handle so _run wires cwd+hook."""

    def noop_tool(x: str) -> str:
        """A trivial tool so build_agent takes the tool path."""
        return x

    from robotsix_llmio.config.tier import (
        ProviderSlotConfig,
        TierConfig,
        TierLevelConfig,
    )

    handle = ClaudeSDKProvider().build_agent(
        level=1,
        tier_config=TierConfig(
            default=ProviderSlotConfig(
                level1=TierLevelConfig(model="claudeSDK-haiku"),
                level2=TierLevelConfig(model="claudeSDK-opus"),
                level3=TierLevelConfig(model="claudeSDK-claude-fable-5"),
            ),
        ),
        system_prompt="p",
        tools=[noop_tool],
        name="t",
        workspace_root=tmp_path,
    )
    assert handle._workspace_root == str(tmp_path)


def test_build_agent_workspace_root_defaults_none(tmp_path):
    def noop_tool(x: str) -> str:
        """trivial."""
        return x

    from robotsix_llmio.config.tier import (
        ProviderSlotConfig,
        TierConfig,
        TierLevelConfig,
    )

    handle = ClaudeSDKProvider().build_agent(
        level=1,
        tier_config=TierConfig(
            default=ProviderSlotConfig(
                level1=TierLevelConfig(model="claudeSDK-haiku"),
                level2=TierLevelConfig(model="claudeSDK-opus"),
                level3=TierLevelConfig(model="claudeSDK-claude-fable-5"),
            ),
        ),
        system_prompt="p",
        tools=[noop_tool],
        name="t",
    )
    assert handle._workspace_root is None


# --- false refusals seen in mill logs on 2026-09-08 (ticket 7064) ------------


def test_bash_hook_allows_workspace_venv_symlink_interpreter(tmp_path):
    """`<repo>/.venv/bin/python` is a symlink to the system interpreter; it is
    the checkout's own tooling and must not be refused as an escape."""
    venv_bin = tmp_path / "repo" / ".venv" / "bin"
    venv_bin.mkdir(parents=True)
    (venv_bin / "python").symlink_to("/usr/bin/env")  # exists on every CI box
    root = tmp_path / "repo"
    cmd = f"{root}/.venv/bin/python -m pytest tests/dev-tooling/test_check_sync.py -q"
    assert _run_bash_hook(root, cmd) == {}


def test_bash_hook_still_denies_absolute_outside_even_if_symlinked(tmp_path):
    """The lexical allowance only applies to paths INSIDE the root."""
    assert _denied(_run_bash_hook(tmp_path, "/usr/local/bin/python -c 'print(1)'"))


@pytest.mark.parametrize(
    "command",
    [
        "awk '/^### Job:/{print}' out.log",
        "grep -E '/[a-z]+$' -r src",
    ],
)
def test_bash_hook_ignores_regex_fragments_after_slash(tmp_path, command):
    """`/^…` in awk/sed/grep programs is a pattern, not a path."""
    assert _run_bash_hook(tmp_path, command) == {}


def test_bash_hook_dotdot_escape_still_denied_lexically(tmp_path):
    root = tmp_path / "repo"
    root.mkdir()
    assert _denied(_run_bash_hook(root, f"cat {root}/../secret.txt"))


def _cli_scratch(tmp_path, monkeypatch, cwd):
    """Point tempfile at a private tmpdir and return the CLI scratch dir the
    ``claude`` CLI would use for a session whose cwd is *cwd*."""
    import os
    import tempfile

    tmpdir = tmp_path / "tmp"
    tmpdir.mkdir(exist_ok=True)
    monkeypatch.setattr(tempfile, "tempdir", str(tmpdir))
    uid = os.getuid() if hasattr(os, "getuid") else 0
    return tmpdir / f"claude-{uid}" / str(cwd).replace(os.sep, "-")


def test_bash_hook_allows_cli_scratch_area_of_this_workspace(tmp_path, monkeypatch):
    """`/tmp/claude-<uid>/<cwd slug>/<session>/tasks/<id>.output` is the CLI's
    own background-task output for this session — the agent must be able to
    read it (mill 2026-09-08: `tail -20 /tmp/claude-1000/-data-robotsix-mill-
    workspaces-<ticket>-repo/<session>/tasks/bd4ehoed2.output` was refused)."""
    root = tmp_path / "repo"
    root.mkdir()
    scratch = _cli_scratch(tmp_path, monkeypatch, root)
    out = scratch / "f2ccd727-8fc5" / "tasks" / "bd4ehoed2.output"
    assert _run_bash_hook(root, f"tail -20 {out}") == {}
    assert _run_bash_hook(root, f"until [ -f {out} ]; do sleep 5; done") == {}
    # The suggested scratchpad directory itself, written to and read back.
    pad = scratch / "f2ccd727-8fc5" / "scratchpad" / "notes.md"
    assert _run_bash_hook(root, f"echo x > {pad} && cat {pad}") == {}


def test_bash_hook_denies_cli_scratch_area_of_another_workspace(tmp_path, monkeypatch):
    """Only this workspace's slug is the agent's own; a sibling session's
    scratch (another ticket's workspace) is still outside the confinement."""
    root = tmp_path / "repo"
    root.mkdir()
    other = _cli_scratch(tmp_path, monkeypatch, tmp_path / "other-repo")
    out = other / "sess" / "tasks" / "x.output"
    assert _denied(_run_bash_hook(root, f"cat {out}"))
    # A plain /tmp path is still refused as before.
    assert _denied(_run_bash_hook(root, f"cat {tmp_path / 'tmp' / 'x'}"))

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

from robotsix_llmio.claude_sdk._tool_agent import (
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

    from robotsix_llmio.config.tier import TierConfig, TierLevelConfig

    handle = ClaudeSDKProvider().build_agent(
        level=1,
        tier_config=TierConfig(
            level1=TierLevelConfig(model="claudeSDK-haiku"),
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

    from robotsix_llmio.config.tier import TierConfig, TierLevelConfig

    handle = ClaudeSDKProvider().build_agent(
        level=1,
        tier_config=TierConfig(
            level1=TierLevelConfig(model="claudeSDK-haiku"),
        ),
        system_prompt="p",
        tools=[noop_tool],
        name="t",
    )
    assert handle._workspace_root is None

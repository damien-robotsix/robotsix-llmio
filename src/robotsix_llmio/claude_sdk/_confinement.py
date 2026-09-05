"""Workspace confinement hooks — path-traversal guards for the SDK tool path.

A tool-bearing claude_sdk agent runs under ``permission_mode="bypassPermissions"``,
so the SDK's built-in ``Write``/``Edit``/``MultiEdit``/``NotebookEdit`` tools can
write anywhere the process can reach.  When ``build_agent(workspace_root=...)`` is
set, these ``PreToolUse`` hooks DENY any edit whose target resolves outside the
workspace while allowing edits inside it.
"""

from __future__ import annotations

import logging
import os
import re
from typing import TYPE_CHECKING, Any, cast

if TYPE_CHECKING:  # pragma: no cover — types-only; runtime imports stay lazy
    from claude_agent_sdk import (
        HookCallback,
    )

log = logging.getLogger("robotsix_llmio.claude_sdk")

# Built-in tools whose input names a file the agent is about to write. The
# hook confines these to the workspace; reads/exploration are left free.
_EDIT_TOOLS = "Write|Edit|MultiEdit|NotebookEdit"
_EDIT_PATH_KEYS = ("file_path", "notebook_path", "path")

# Pseudo-devices that are safe to reference from a confined Bash command:
# they neither read nor leak workspace-external data. `/dev/fd/N` (process
# substitution) is allowed by prefix in the hook.
_SAFE_PSEUDO_DEVICES = frozenset(
    {
        "/dev/null",
        "/dev/stdin",
        "/dev/stdout",
        "/dev/stderr",
        "/dev/tty",
        "/dev/zero",
        "/dev/urandom",
        "/dev/random",
    }
)


def _is_within(root: str, target: str) -> bool:
    """True if *target* (resolved, relative paths joined to *root*) is inside
    *root*. ``realpath`` collapses ``..`` and symlinks so escapes are caught."""
    p = target if os.path.isabs(target) else os.path.join(root, target)
    rp = os.path.realpath(p)
    return rp == root or rp.startswith(root + os.sep)


def _deny_hook_output(reason: str) -> dict[str, Any]:
    """Return a PreToolUse deny decision dict for the given *reason* string."""
    return {
        "hookSpecificOutput": {
            "hookEventName": "PreToolUse",
            "permissionDecision": "deny",
            "permissionDecisionReason": reason,
        }
    }


def _make_confine_hook(workspace_root: str) -> HookCallback:
    """Build a ``PreToolUse`` hook that denies built-in edits outside
    *workspace_root*.

    ``permission_mode="bypassPermissions"`` lets the SDK's built-in
    Write/Edit/etc. write anywhere the process can reach, so a tool-bearing
    agent working on a self-referential ticket can edit the host app's own
    source instead of its checkout. A PreToolUse hook is the one gate the SDK
    consults on *every* call regardless of permission mode (``can_use_tool``
    is skipped under bypass), so it is where confinement must live."""
    root = os.path.realpath(workspace_root)

    async def _hook(
        input: dict[str, Any], tool_use_id: str | None, context: Any
    ) -> dict[str, Any]:
        tool_input = input.get("tool_input") or {}
        target = next(
            (tool_input[k] for k in _EDIT_PATH_KEYS if tool_input.get(k)), None
        )
        if not target or _is_within(root, str(target)):
            return {}  # no path, or inside the workspace → allow
        log.warning(
            "%s: denied out-of-workspace edit to %s (confined to %s)",
            input.get("tool_name", "edit"),
            target,
            root,
        )
        return _deny_hook_output(
            f"Refused: edits are confined to the ticket workspace "
            f"{root}. {target!r} resolves outside it — edit the "
            f"corresponding file inside the workspace checkout instead."
        )

    return cast("HookCallback", _hook)


def _make_bash_confine_hook(workspace_root: str) -> HookCallback:
    """PreToolUse hook that denies Bash commands naming absolute paths outside
    *workspace_root*.

    Parsing is heuristic: tokens that start with ``/`` (absolute paths) are
    extracted and resolved. Commands that construct paths at runtime via
    subshells, ``eval``, or base64 encoding are NOT caught — this is
    documented by design."""
    root = os.path.realpath(workspace_root)

    async def _hook(
        input: dict[str, Any], tool_use_id: str | None, context: Any
    ) -> dict[str, Any]:
        tool_input = input.get("tool_input") or {}
        command = str(tool_input.get("command") or "")
        if not command:
            return {}
        # Extract absolute-path-like tokens: sequences of non-whitespace chars
        # starting with / after a word boundary (whitespace, operator, or BOL).
        for match in re.finditer(
            r"(?:(?<=\s)|(?<=^)|(?<=[;|&><\x27\x22=({!,`]))"
            r"(/[^\s\x27\x22\\;|&><`)}]+)",
            command,
        ):
            candidate = match.group(1).rstrip("'\";)}")  # strip trailing punct
            if candidate in _SAFE_PSEUDO_DEVICES or candidate.startswith("/dev/fd/"):
                # `2>/dev/null` and friends are ubiquitous shell idioms that
                # neither read nor leak anything outside the workspace;
                # denying them burned review/ci_fix agent turns on false
                # refusals (mill 2026-09-05).
                continue
            if candidate and not _is_within(root, candidate):
                log.warning(
                    "Bash: denied out-of-workspace path %s (confined to %s)",
                    candidate,
                    root,
                )
                return _deny_hook_output(
                    f"Refused: Bash command references {candidate!r}, "
                    f"which resolves outside the confined workspace "
                    f"{root}. Use paths inside the workspace checkout instead."
                )
        return {}

    return cast("HookCallback", _hook)

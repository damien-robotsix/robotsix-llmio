"""Tests for the argv-limit system-prompt spill (``_system_prompt``).

Live incident 2026-09-05: a 91 KB ``agent_instruction`` made the CLI spawn
fail with ``[Errno 7] Argument list too long`` because the SDK passes
``system_prompt`` as one ``--system-prompt <text>`` argv element and Linux
caps a single argv string at 128 KiB.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field

from robotsix_llmio.claude_sdk._system_prompt import (
    _ARGV_SAFE_BYTES,
    spill_oversized_system_prompt,
)


@dataclass
class _Options:
    """The two ``ClaudeAgentOptions`` fields the spill touches, real shapes:
    ``system_prompt`` is ``str | None`` (or the SDK preset dict) and
    ``extra_args`` defaults to an empty dict mapping flag names to values."""

    system_prompt: object = None
    extra_args: dict = field(default_factory=dict)


def test_small_prompt_left_on_argv():
    opts = _Options(system_prompt="short instruction")
    cleanup = spill_oversized_system_prompt(opts)
    assert opts.system_prompt == "short instruction"
    assert "system-prompt-file" not in opts.extra_args
    cleanup()  # no-op


def test_none_and_preset_dict_left_untouched():
    for value in (None, {"type": "preset", "preset": "claude_code"}):
        opts = _Options(system_prompt=value)
        cleanup = spill_oversized_system_prompt(opts)
        assert opts.system_prompt == value
        assert opts.extra_args == {}
        cleanup()


def test_oversized_prompt_spills_to_file_and_cleans_up():
    big = "x" * (_ARGV_SAFE_BYTES + 1)
    opts = _Options(system_prompt=big, extra_args={"existing-flag": "kept"})
    cleanup = spill_oversized_system_prompt(opts)

    assert opts.system_prompt is None
    path = opts.extra_args["system-prompt-file"]
    assert opts.extra_args["existing-flag"] == "kept"
    with open(path, encoding="utf-8") as fh:
        assert fh.read() == big

    cleanup()
    assert not os.path.exists(path)
    cleanup()  # idempotent


def test_threshold_counts_utf8_bytes_not_chars():
    # é is 2 UTF-8 bytes: half the cap in characters is over it in bytes.
    big = "é" * (_ARGV_SAFE_BYTES // 2 + 1)
    opts = _Options(system_prompt=big)
    cleanup = spill_oversized_system_prompt(opts)
    assert opts.system_prompt is None
    path = opts.extra_args["system-prompt-file"]
    with open(path, encoding="utf-8") as fh:
        assert fh.read() == big
    cleanup()

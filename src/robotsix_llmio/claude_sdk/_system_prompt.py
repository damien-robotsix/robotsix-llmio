"""Deliver oversized system prompts to the claude CLI via a file, not argv.

The SDK passes ``ClaudeAgentOptions.system_prompt`` to the bundled CLI as a
single ``--system-prompt <text>`` argv element.  Linux caps one argv string at
``MAX_ARG_STRLEN`` (128 KiB), so a large system prompt aborts the spawn with
``[Errno 7] Argument list too long`` (E2BIG) — a laundered transport failure
the caller cannot decode.  Live incident 2026-09-05: robotsix-chat's 91 KB
default ``agent_instruction`` plus per-call additions crossed the cap and
every chat turn failed until the prompt was shrunk.

The CLI accepts ``--system-prompt-file <path>`` as an equivalent out-of-band
channel; :func:`spill_oversized_system_prompt` switches to it whenever the
prompt is anywhere near the argv cap.
"""

from __future__ import annotations

import contextlib
import logging
import os
import tempfile
from collections.abc import Callable
from typing import TYPE_CHECKING

if TYPE_CHECKING:  # pragma: no cover
    from claude_agent_sdk import ClaudeAgentOptions

log = logging.getLogger(__name__)

# Stay well below MAX_ARG_STRLEN (131072): the exec budget is shared with every
# other argv element and environment string, and the kernel counts the
# terminating NUL and pointer overhead too.
_ARGV_SAFE_BYTES = 65536


def spill_oversized_system_prompt(
    options: ClaudeAgentOptions,
) -> Callable[[], None]:
    """Move an argv-unsafe ``system_prompt`` into a ``--system-prompt-file``.

    When ``options.system_prompt`` is a string whose UTF-8 encoding exceeds
    ``_ARGV_SAFE_BYTES``, write it to a private temp file, clear
    ``options.system_prompt`` and point the CLI at the file through
    ``options.extra_args["system-prompt-file"]``.  Smaller prompts (and the
    SDK's preset-dict form) are left untouched.

    Returns:
        A cleanup callable that removes the temp file (a no-op when nothing
        was spilled).  Call it after the SDK query completes — the CLI reads
        the file once at startup.

    """
    system_prompt = options.system_prompt
    if not isinstance(system_prompt, str):
        return lambda: None
    encoded = system_prompt.encode("utf-8")
    if len(encoded) <= _ARGV_SAFE_BYTES:
        return lambda: None

    fd, path = tempfile.mkstemp(prefix="llmio-system-prompt-", suffix=".md")
    try:
        with os.fdopen(fd, "wb") as fh:
            fh.write(encoded)
    except OSError:
        os.unlink(path)
        raise
    log.info(
        "system prompt is %d bytes (> %d argv-safe): passing via "
        "--system-prompt-file %s",
        len(encoded),
        _ARGV_SAFE_BYTES,
        path,
    )
    options.system_prompt = None
    options.extra_args = {**options.extra_args, "system-prompt-file": path}

    def _cleanup() -> None:
        with contextlib.suppress(OSError):
            os.unlink(path)

    return _cleanup

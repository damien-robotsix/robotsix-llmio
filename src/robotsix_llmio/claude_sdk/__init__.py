"""Claude Agent SDK transport layer (subscription / ``claude login`` auth).

Requires the ``claude_sdk`` extra plus a logged-in ``claude`` CLI and Node.js at
runtime. The model/provider are loaded lazily via PEP 562 ``__getattr__`` so
importing the lightweight ``transient`` helpers stays free of the SDK; a missing
extra surfaces a clear install hint when the model/provider is actually used.
"""

from __future__ import annotations

from typing import Any

from ._stream import ClaudeSDKActivityEvent, activity_events
from .transient import (
    is_claude_sdk_transient,
    is_claude_sdk_turn_limit,
    is_claude_sdk_usage_exhausted,
)

__all__ = [
    "ClaudeSDKAPIError",
    "ClaudeSDKActivityEvent",
    "ClaudeSDKModel",
    "ClaudeSDKProvider",
    "ClaudeSDKQueryTimeout",
    "ClaudeSDKTurnLimitError",
    "ClaudeSDKUsageExhaustedError",
    "activity_events",
    "is_claude_sdk_transient",
    "is_claude_sdk_turn_limit",
    "is_claude_sdk_usage_exhausted",
]


def __getattr__(name: str) -> Any:  # PEP 562 — lazy heavy imports
    if name in (
        "ClaudeSDKProvider",
        "ClaudeSDKModel",
        "ClaudeSDKAPIError",
        "ClaudeSDKTurnLimitError",
        "ClaudeSDKQueryTimeout",
        "ClaudeSDKUsageExhaustedError",
    ):
        try:
            if name == "ClaudeSDKProvider":
                from .provider import ClaudeSDKProvider

                return ClaudeSDKProvider
            if name == "ClaudeSDKTurnLimitError":
                from ._errors import ClaudeSDKTurnLimitError

                return ClaudeSDKTurnLimitError
            if name == "ClaudeSDKQueryTimeout":
                from ._errors import ClaudeSDKQueryTimeout

                return ClaudeSDKQueryTimeout
            if name == "ClaudeSDKUsageExhaustedError":
                from ._errors import ClaudeSDKUsageExhaustedError

                return ClaudeSDKUsageExhaustedError
            if name == "ClaudeSDKAPIError":
                from ._errors import ClaudeSDKAPIError

                return ClaudeSDKAPIError
            from ._model import ClaudeSDKModel

            return ClaudeSDKModel
        except ImportError as exc:  # pragma: no cover
            raise ImportError(
                "robotsix_llmio.claude_sdk requires the 'claude_sdk' extra. "
                "Install with: pip install 'robotsix-llmio[claude_sdk]' "
                "(also needs Node.js and a logged-in `claude` CLI)."
            ) from exc
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

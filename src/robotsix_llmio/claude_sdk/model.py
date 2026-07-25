"""Re-export shim.

All public and internal symbols are defined in the private sub-modules:

* :mod:`~robotsix_llmio.claude_sdk._errors` — exception classes
* :mod:`~robotsix_llmio.claude_sdk._prompt` — prompt/image helpers
* :mod:`~robotsix_llmio.claude_sdk._model` — ``ClaudeSDKModel``, constants
"""

from ._errors import (  # re-exported for backward compat
    ClaudeSDKAPIError,
    ClaudeSDKQueryTimeout,
    ClaudeSDKTurnLimitError,
    ClaudeSDKUsageExhaustedError,
)
from ._model import (  # noqa: F401  # re-exported for backward compat
    _MAX_TURNS,
    PROVIDER_NAME,
    ClaudeSDKModel,
)
from ._prompt import (  # noqa: F401  # re-exported for backward compat
    _map_usage,
    build_sdk_prompt,
    collect_latest_user_images,
    extract_prompt_parts,
    render_prompt,
)

__all__ = [
    "ClaudeSDKAPIError",
    "ClaudeSDKModel",
    "ClaudeSDKQueryTimeout",
    "ClaudeSDKTurnLimitError",
    "ClaudeSDKUsageExhaustedError",
]

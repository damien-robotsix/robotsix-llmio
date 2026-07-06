Extend usage-credit-exhaustion detection to the case where `claude_agent_sdk` collapses the
condition into a raised generic exception (e.g. the degenerate-success message) instead of a clean
`is_error=True` return, discarding the real "You're out of usage credits" text before it would
otherwise be checked. `_stream_query` now also checks the assistant text already streamed into
`chunks` before such an exception fires, so this case raises `ClaudeSDKUsageExhaustedError` too
instead of being misclassified as an ordinary transient error, retried 3x at the same exhausted
tier, and finally leaked to the caller verbatim.

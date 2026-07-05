Detect Claude subscription usage-credit exhaustion as a distinct failure. The SDK reports this as
a normal-looking `is_error=True` result carrying the text "You're out of usage credits" rather than
raising, so it was previously returned as if it were a genuine reply, or (when the SDK did raise)
retried 3x at the same exhausted tier as a misclassified "degenerate success". `_stream_query` now
raises the new `ClaudeSDKUsageExhaustedError` for this specific signature, and
`is_claude_sdk_transient` excludes it (like the existing turn-limit case) so it fails immediately
instead of burning retries at a tier that cannot recover until its credits reset.

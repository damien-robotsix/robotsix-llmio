# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- Root exception class `RobotsixLLMIOError` that all library-specific errors inherit from, allowing callers to catch library exceptions with a single `except` clause. Exported from top-level package.

### Changed
- **Breaking**: `ClaudeSDKTurnLimitError` and `ClaudeSDKQueryTimeout` now inherit from `RobotsixLLMIOError` instead of `RuntimeError` and `TimeoutError` respectively. Callers catching these exceptions by type must update to catch `RobotsixLLMIOError`.

## [0.1.0] - 2026-06-13

Initial release.

"""Tests for untested private modules in ``src/robotsix_llmio/core/``.

- ``_output_markers._is_output_type_marked`` — marker-type detection
- ``_rest_client._DEFAULT_BASE_URL`` — pin the literal value
"""

from __future__ import annotations

from pydantic import BaseModel
from pydantic_ai import NativeOutput, PromptedOutput, ToolOutput

from robotsix_llmio.core._output_markers import _is_output_type_marked
from robotsix_llmio.core._rest_client import _DEFAULT_BASE_URL

# ---------------------------------------------------------------------------
# _rest_client
# ---------------------------------------------------------------------------


def test_default_base_url_value() -> None:
    """The constant MUST be the expected URL — consumers rely on it."""
    assert _DEFAULT_BASE_URL == "http://localhost:8000/api/v1"


# ---------------------------------------------------------------------------
# _is_output_type_marked
# ---------------------------------------------------------------------------


def test_is_output_type_marked_for_marker_instances() -> None:
    """Returns True for PromptedOutput, ToolOutput, NativeOutput instances."""
    assert _is_output_type_marked(PromptedOutput(str)) is True
    assert _is_output_type_marked(ToolOutput(str)) is True
    assert _is_output_type_marked(NativeOutput(str)) is True


def test_is_output_type_marked_for_list_of_markers() -> None:
    """Returns True for a list containing a marker type."""
    assert _is_output_type_marked([PromptedOutput(str)]) is True


def test_is_output_type_marked_for_tuple_of_markers() -> None:
    """Returns True for a tuple containing a marker type."""
    assert _is_output_type_marked((NativeOutput(str),)) is True


def test_is_output_type_marked_for_mixed_list() -> None:
    """Returns True for a list containing both marker and non-marker types."""
    assert _is_output_type_marked([str, NativeOutput(str)]) is True


def test_is_output_type_marked_negative_plain_types() -> None:
    """Returns False for str, int, and other arbitrary types."""
    assert _is_output_type_marked(str) is False
    assert _is_output_type_marked(int) is False
    assert _is_output_type_marked(float) is False


def test_is_output_type_marked_negative_base_model() -> None:
    """Returns False for a plain pydantic BaseModel subclass."""

    class MyModel(BaseModel):
        x: int

    assert _is_output_type_marked(MyModel) is False


def test_is_output_type_marked_negative_empty_list() -> None:
    """Returns False for an empty list."""
    assert _is_output_type_marked([]) is False


def test_is_output_type_marked_negative_empty_tuple() -> None:
    """Returns False for an empty tuple."""
    assert _is_output_type_marked(()) is False


def test_is_output_type_marked_negative_plain_list() -> None:
    """Returns False for a list of non-marker types."""
    assert _is_output_type_marked([str, int]) is False

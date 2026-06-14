"""Tests for the dependency-free ``html_to_text`` helper."""

from __future__ import annotations

from robotsix_llmio.core import html_to_text


def test_script_block_content_removed() -> None:
    assert (
        html_to_text("<p>Hi &amp; <b>bye</b></p><script>alert(1)</script>")
        == "Hi & bye"
    )


def test_style_block_content_removed() -> None:
    assert html_to_text("<style>p{color:red}</style><p>text</p>") == "text"


def test_noscript_block_content_removed() -> None:
    assert html_to_text("<noscript>fallback</noscript><p>real</p>") == "real"


def test_svg_block_content_removed() -> None:
    assert html_to_text("<svg><circle r='5'/></svg><p>after</p>") == "after"


def test_drop_blocks_case_insensitive() -> None:
    assert html_to_text("<SCRIPT>alert(1)</SCRIPT>keep") == "keep"


def test_generic_tag_stripping() -> None:
    assert html_to_text("<div><span>hello</span></div>") == "hello"


def test_entity_unescaping() -> None:
    # ``&amp;`` and numeric ``&#38;`` decode to ``&``; ``&nbsp;`` decodes to a
    # non-breaking space which the whitespace pass then collapses to a space.
    assert html_to_text("a &amp; b&nbsp;c &#38; d") == "a & b c & d"


def test_whitespace_collapse_and_trim() -> None:
    assert html_to_text("  plain   text\n\there  ") == "plain text here"


def test_empty_string() -> None:
    assert html_to_text("") == ""


def test_plaintext_passthrough() -> None:
    assert html_to_text("just plain text") == "just plain text"


def test_nested_and_attribute_bearing_tags() -> None:
    assert (
        html_to_text('<a href="https://x.test"><em>link</em> here</a>') == "link here"
    )

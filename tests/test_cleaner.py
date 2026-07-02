import pytest
from src.ingestion.cleaner import clean
from src.ingestion.models import Document


def _txt_doc(content: str) -> Document:
    return Document(content=content, source="test.txt", file_type="txt")


def _md_doc(content: str) -> Document:
    return Document(content=content, source="test.md", file_type="md")


# ── Return type and field preservation ───────────────────────────────────────

def test_clean_returns_document():
    assert isinstance(clean(_txt_doc("hello")), Document)


def test_clean_preserves_source():
    doc = _txt_doc("hello")
    assert clean(doc).source == doc.source


def test_clean_preserves_file_type():
    doc = _txt_doc("hello")
    assert clean(doc).file_type == doc.file_type


def test_clean_preserves_page():
    doc = Document(content="hello", source="f.txt", file_type="txt", page=3)
    assert clean(doc).page == 3


def test_clean_preserves_metadata():
    doc = Document(content="hello", source="f.txt", file_type="txt", metadata={"k": "v"})
    assert clean(doc).metadata == {"k": "v"}


# ── TXT: whitespace normalisation ─────────────────────────────────────────────

def test_txt_strips_trailing_whitespace():
    result = clean(_txt_doc("hello   \nworld   "))
    assert result.content == "hello\nworld"


def test_txt_normalises_crlf():
    result = clean(_txt_doc("line1\r\nline2"))
    assert "\r" not in result.content


def test_txt_collapses_multiple_blank_lines():
    result = clean(_txt_doc("a\n\n\n\n\nb"))
    assert result.content == "a\n\nb"


def test_txt_strips_leading_and_trailing_blank_lines():
    result = clean(_txt_doc("\n\nhello\n\n"))
    assert result.content == "hello"


def test_txt_preserves_single_blank_line():
    result = clean(_txt_doc("a\n\nb"))
    assert result.content == "a\n\nb"


# ── Markdown: heading stripping ───────────────────────────────────────────────

def test_md_strips_h1():
    result = clean(_md_doc("# My Title"))
    assert result.content == "My Title"


def test_md_strips_h2():
    result = clean(_md_doc("## Section"))
    assert result.content == "Section"


def test_md_strips_h6():
    result = clean(_md_doc("###### Deep"))
    assert result.content == "Deep"


# ── Markdown: inline formatting ───────────────────────────────────────────────

def test_md_strips_bold_asterisk():
    result = clean(_md_doc("This is **bold** text."))
    assert result.content == "This is bold text."


def test_md_strips_bold_underscore():
    result = clean(_md_doc("This is __bold__ text."))
    assert result.content == "This is bold text."


def test_md_strips_italic_asterisk():
    result = clean(_md_doc("This is *italic* text."))
    assert result.content == "This is italic text."


def test_md_strips_inline_code():
    result = clean(_md_doc("Use `print()` to output."))
    assert result.content == "Use print() to output."


# ── Markdown: links and images ────────────────────────────────────────────────

def test_md_keeps_link_text():
    result = clean(_md_doc("[OpenAI](https://openai.com)"))
    assert result.content == "OpenAI"


def test_md_removes_images():
    result = clean(_md_doc("Here is ![a diagram](diagram.png) shown."))
    assert "diagram.png" not in result.content
    assert "![" not in result.content


# ── Markdown: code blocks ─────────────────────────────────────────────────────

def test_md_removes_fenced_code_delimiters():
    md = "Example:\n```python\nprint('hello')\n```\nDone."
    result = clean(_md_doc(md))
    assert "```" not in result.content


def test_md_keeps_code_block_content():
    md = "```\nsome code\n```"
    result = clean(_md_doc(md))
    assert "some code" in result.content


# ── Markdown: tables ─────────────────────────────────────────────────────────

def test_md_removes_table_separator_rows():
    md = "| A | B |\n|---|---|\n| 1 | 2 |"
    result = clean(_md_doc(md))
    assert "|---|" not in result.content


# ── Markdown: horizontal rules ────────────────────────────────────────────────

def test_md_removes_horizontal_rules():
    result = clean(_md_doc("Before\n---\nAfter"))
    assert "---" not in result.content


# ── Integration: sample files ─────────────────────────────────────────────────

def test_clean_sample_txt_has_no_excess_blank_lines():
    from pathlib import Path
    from src.ingestion.loader import load
    docs = load(Path("data/sample.txt"))
    cleaned = clean(docs[0])
    lines = cleaned.content.split("\n")
    for i in range(len(lines) - 1):
        assert not (lines[i] == "" and lines[i + 1] == ""), \
            f"Found consecutive blank lines at line {i}"


def test_clean_sample_md_has_no_markdown_headings():
    from pathlib import Path
    from src.ingestion.loader import load
    docs = load(Path("data/sample.md"))
    cleaned = clean(docs[0])
    for line in cleaned.content.split("\n"):
        assert not line.startswith("#"), f"Heading not stripped: {line!r}"

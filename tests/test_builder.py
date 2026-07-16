"""Tests for the prompt builder — pure string logic, no I/O needed."""
import pytest
from src.prompt.builder import build, format_sources
from src.prompt.templates import SYSTEM_PROMPT, NO_CONTEXT_PLACEHOLDER
from src.ingestion.models import Chunk


def _chunk(
    text: str = "Sample chunk text.",
    source: str = "/data/report.txt",
    page: int = 1,
) -> Chunk:
    return Chunk(
        text=text, source=source, file_type="txt",
        page=page, chunk_index=0, start_char=0, end_char=len(text),
    )


# ── Return type ───────────────────────────────────────────────────────────────

def test_build_returns_string():
    assert isinstance(build([_chunk()], "What is RAG?"), str)


def test_build_returns_non_empty_string():
    assert len(build([_chunk()], "What is RAG?")) > 0


# ── System prompt ─────────────────────────────────────────────────────────────

def test_build_includes_system_prompt():
    result = build([_chunk()], "question")
    assert SYSTEM_PROMPT in result


# ── Question ──────────────────────────────────────────────────────────────────

def test_build_includes_the_question():
    question = "What is Retrieval-Augmented Generation?"
    result = build([_chunk()], question)
    assert question in result


def test_build_includes_answer_marker():
    result = build([_chunk()], "question")
    assert "Answer:" in result


def test_build_question_appears_after_context():
    result = build([_chunk(text="context text")], "my question")
    assert result.index("context text") < result.index("my question")


# ── Chunk content ─────────────────────────────────────────────────────────────

def test_build_includes_chunk_text():
    result = build([_chunk(text="RAG stands for Retrieval-Augmented Generation.")], "q")
    assert "RAG stands for Retrieval-Augmented Generation." in result


def test_build_includes_source_filename_not_full_path():
    result = build([_chunk(source="/very/long/path/report.txt")], "q")
    assert "report.txt" in result
    assert "/very/long/path/" not in result


def test_build_includes_page_number():
    result = build([_chunk(page=7)], "q")
    assert "page 7" in result


def test_build_numbers_first_chunk_as_1():
    result = build([_chunk()], "q")
    assert "[1]" in result


def test_build_numbers_chunks_sequentially():
    chunks = [_chunk(text=f"chunk {i}") for i in range(3)]
    result = build(chunks, "q")
    assert "[1]" in result
    assert "[2]" in result
    assert "[3]" in result


def test_build_all_chunk_texts_present():
    chunks = [
        _chunk(text="First passage about RAG."),
        _chunk(text="Second passage about FAISS."),
        _chunk(text="Third passage about embeddings."),
    ]
    result = build(chunks, "q")
    assert "First passage about RAG." in result
    assert "Second passage about FAISS." in result
    assert "Third passage about embeddings." in result


def test_build_chunk_rank_order_matches_list_order():
    chunks = [_chunk(text="alpha"), _chunk(text="beta")]
    result = build(chunks, "q")
    assert result.index("[1]") < result.index("[2]")
    assert result.index("alpha") < result.index("beta")


def test_build_different_sources_shown_separately():
    chunks = [
        _chunk(text="from report", source="/data/report.txt", page=1),
        _chunk(text="from guide", source="/data/guide.md", page=3),
    ]
    result = build(chunks, "q")
    assert "report.txt" in result
    assert "guide.md" in result
    assert "page 1" in result
    assert "page 3" in result


# ── Empty chunks ──────────────────────────────────────────────────────────────

def test_build_empty_chunks_uses_no_context_placeholder():
    result = build([], "What is RAG?")
    assert NO_CONTEXT_PLACEHOLDER in result


def test_build_empty_chunks_still_includes_question():
    result = build([], "What is RAG?")
    assert "What is RAG?" in result


def test_build_empty_chunks_still_includes_answer_marker():
    result = build([], "question")
    assert "Answer:" in result


# ── Error handling ────────────────────────────────────────────────────────────

def test_build_empty_question_raises_value_error():
    with pytest.raises(ValueError, match="non-empty"):
        build([_chunk()], "")


def test_build_whitespace_question_raises_value_error():
    with pytest.raises(ValueError):
        build([_chunk()], "   ")


# ── Structure ─────────────────────────────────────────────────────────────────

def test_build_system_prompt_appears_before_context():
    result = build([_chunk(text="context here")], "question")
    assert result.index(SYSTEM_PROMPT) < result.index("context here")


def test_build_context_appears_before_question():
    result = build([_chunk(text="the context")], "the question")
    assert result.index("the context") < result.index("the question")


def test_build_single_chunk_no_stray_numbering():
    result = build([_chunk()], "q")
    assert "[2]" not in result
    assert "[0]" not in result


# ── format_sources() — return type ─────────────────────────────────────────────

def test_format_sources_returns_string():
    assert isinstance(format_sources([_chunk()]), str)


def test_format_sources_starts_with_header():
    result = format_sources([_chunk()])
    assert result.startswith("Sources:")


# ── format_sources() — one source ───────────────────────────────────────────────

def test_format_sources_single_source_shows_filename():
    result = format_sources([_chunk(source="/data/report.txt")])
    assert "report.txt" in result


def test_format_sources_single_source_shows_page():
    result = format_sources([_chunk(page=3)])
    assert "page 3" in result


def test_format_sources_single_source_uses_filename_not_full_path():
    result = format_sources([_chunk(source="/very/long/path/report.txt")])
    assert "report.txt" in result
    assert "/very/long/path/" not in result


def test_format_sources_single_source_numbered_1():
    result = format_sources([_chunk()])
    assert "[1]" in result


# ── format_sources() — deduplication within one document ────────────────────────

def test_format_sources_dedupes_multiple_chunks_same_source_and_page():
    chunks = [
        _chunk(text="first", source="/data/report.txt", page=1),
        _chunk(text="second", source="/data/report.txt", page=1),
        _chunk(text="third", source="/data/report.txt", page=1),
    ]
    result = format_sources(chunks)
    assert result.count("report.txt") == 1


def test_format_sources_does_not_dedupe_different_pages_of_same_document():
    chunks = [
        _chunk(source="/data/report.pdf", page=1),
        _chunk(source="/data/report.pdf", page=2),
    ]
    result = format_sources(chunks)
    assert "page 1" in result
    assert "page 2" in result
    assert result.count("report.pdf") == 2


# ── format_sources() — multiple documents ────────────────────────────────────────

def test_format_sources_shows_multiple_different_documents():
    chunks = [
        _chunk(source="/data/report.txt", page=1),
        _chunk(source="/data/guide.md", page=1),
    ]
    result = format_sources(chunks)
    assert "report.txt" in result
    assert "guide.md" in result


def test_format_sources_numbers_dedupe_sequentially_not_by_raw_rank():
    """Three chunks from the same doc + one from another must number [1], [2] — not [1], [4]."""
    chunks = [
        _chunk(source="/data/report.txt", page=1),
        _chunk(source="/data/report.txt", page=1),
        _chunk(source="/data/report.txt", page=1),
        _chunk(source="/data/guide.md", page=1),
    ]
    result = format_sources(chunks)
    assert "[1]" in result
    assert "[2]" in result
    assert "[3]" not in result


def test_format_sources_order_matches_first_occurrence_relevance_order():
    chunks = [
        _chunk(source="/data/best.txt", page=1),
        _chunk(source="/data/second.txt", page=1),
    ]
    result = format_sources(chunks)
    assert result.index("best.txt") < result.index("second.txt")


# ── format_sources() — missing/unusual metadata does not crash ──────────────────

def test_format_sources_none_page_does_not_crash():
    chunk = _chunk(page=None)
    result = format_sources([chunk])
    assert "report.txt" in result


def test_format_sources_empty_source_does_not_crash():
    chunk = _chunk(source="")
    result = format_sources([chunk])
    assert isinstance(result, str)


# ── format_sources() — empty chunk list ──────────────────────────────────────────

def test_format_sources_empty_list_does_not_crash():
    result = format_sources([])
    assert isinstance(result, str)


def test_format_sources_empty_list_says_none():
    result = format_sources([])
    assert "none" in result.lower()

"""Tests for the prompt builder — pure string logic, no I/O needed."""
import pytest
from src.prompt.builder import build
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

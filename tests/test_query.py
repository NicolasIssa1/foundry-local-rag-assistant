"""
Tests for the query pipeline — src/pipeline/query.py.

Uses a real VectorIndex + ChunkStore written to tmp_path (matching what
'python main.py index' produces on disk), with the embedder and Foundry
runtime mocked so no Foundry Local installation is required.
"""
import pytest
from pathlib import Path
from unittest.mock import MagicMock

from src.pipeline.query import query, QueryResult
from src.vectorstore.index import VectorIndex
from src.vectorstore.store import ChunkStore
from src.ingestion.models import Chunk

DIM = 4
V0 = [1.0, 0.0, 0.0, 0.0]
V1 = [0.0, 1.0, 0.0, 0.0]
V2 = [0.0, 0.0, 1.0, 0.0]


def _mock_embedder(return_vector: list[float]) -> MagicMock:
    embedder = MagicMock()
    embedder.embed_one.return_value = return_vector
    return embedder


def _mock_runtime(answer: str = "The answer.") -> MagicMock:
    runtime = MagicMock()
    runtime.stream_chat.return_value = iter(list(answer))
    runtime.chat.return_value = answer
    return runtime


def _build_index_dir(tmp_path: Path, chunks: list[Chunk], vectors: list[list[float]]) -> Path:
    """Write a FAISS index + SQLite store to tmp_path, matching on-disk layout."""
    index_dir = tmp_path / "index"
    index_dir.mkdir()

    store = ChunkStore(index_dir / "chunks.db")
    store.add(chunks)
    store.close()

    faiss_index = VectorIndex(DIM)
    faiss_index.add(vectors)
    faiss_index.save(index_dir / "faiss.index")

    return index_dir


def _chunk(text: str, source: str, page: int = 1, chunk_index: int = 0) -> Chunk:
    return Chunk(
        text=text, source=source, file_type="txt",
        page=page, chunk_index=chunk_index, start_char=0, end_char=len(text),
    )


# ── Existing behaviour preserved ─────────────────────────────────────────────────

def test_query_returns_queryresult(tmp_path):
    chunks = [_chunk("RAG text.", "/data/report.txt")]
    index_dir = _build_index_dir(tmp_path, chunks, [V0])
    result = query(
        question="What is RAG?", index_dir=index_dir,
        embedder=_mock_embedder(V0), runtime=_mock_runtime("hi"),
        verbose=False,
    )
    assert isinstance(result, QueryResult)


def test_query_answer_matches_streamed_tokens(tmp_path, capsys):
    chunks = [_chunk("RAG text.", "/data/report.txt")]
    index_dir = _build_index_dir(tmp_path, chunks, [V0])
    result = query(
        question="What is RAG?", index_dir=index_dir,
        embedder=_mock_embedder(V0), runtime=_mock_runtime("hello"),
        verbose=False,
    )
    assert result.answer == "hello"


def test_query_streams_answer_tokens_to_stdout(tmp_path, capsys):
    chunks = [_chunk("RAG text.", "/data/report.txt")]
    index_dir = _build_index_dir(tmp_path, chunks, [V0])
    query(
        question="What is RAG?", index_dir=index_dir,
        embedder=_mock_embedder(V0), runtime=_mock_runtime("hello"),
        verbose=False,
    )
    captured = capsys.readouterr()
    assert "hello" in captured.out


def test_query_non_stream_mode_still_returns_answer(tmp_path):
    chunks = [_chunk("RAG text.", "/data/report.txt")]
    index_dir = _build_index_dir(tmp_path, chunks, [V0])
    result = query(
        question="What is RAG?", index_dir=index_dir,
        embedder=_mock_embedder(V0), runtime=_mock_runtime("non-streamed answer"),
        stream=False, verbose=False,
    )
    assert result.answer == "non-streamed answer"


def test_query_raises_file_not_found_when_no_index(tmp_path):
    with pytest.raises(FileNotFoundError):
        query(
            question="What is RAG?", index_dir=tmp_path / "missing",
            embedder=_mock_embedder(V0), runtime=_mock_runtime(), verbose=False,
        )


# ── Sources section — printed to stdout ──────────────────────────────────────────

def test_query_prints_a_sources_section(tmp_path, capsys):
    chunks = [_chunk("RAG text.", "/data/report.txt")]
    index_dir = _build_index_dir(tmp_path, chunks, [V0])
    query(
        question="What is RAG?", index_dir=index_dir,
        embedder=_mock_embedder(V0), runtime=_mock_runtime("hi"),
        verbose=False,
    )
    captured = capsys.readouterr()
    assert "Sources:" in captured.out


def test_query_sources_printed_even_when_verbose_is_false(tmp_path, capsys):
    """Sources must always show, unlike the [query] debug log which is verbose-gated."""
    chunks = [_chunk("RAG text.", "/data/report.txt")]
    index_dir = _build_index_dir(tmp_path, chunks, [V0])
    query(
        question="What is RAG?", index_dir=index_dir,
        embedder=_mock_embedder(V0), runtime=_mock_runtime("hi"),
        verbose=False,
    )
    captured = capsys.readouterr()
    assert "[query] Retrieved" not in captured.out
    assert "Sources:" in captured.out


def test_query_sources_shows_single_source_filename(tmp_path, capsys):
    chunks = [_chunk("RAG text.", "/data/report.txt")]
    index_dir = _build_index_dir(tmp_path, chunks, [V0])
    query(
        question="What is RAG?", index_dir=index_dir,
        embedder=_mock_embedder(V0), runtime=_mock_runtime("hi"),
        verbose=False,
    )
    captured = capsys.readouterr()
    assert "report.txt" in captured.out


def test_query_sources_deduplicate_multiple_chunks_from_same_document(tmp_path, capsys):
    chunks = [
        _chunk("Passage A.", "/data/report.txt", chunk_index=0),
        _chunk("Passage B.", "/data/report.txt", chunk_index=1),
        _chunk("Passage C.", "/data/report.txt", chunk_index=2),
    ]
    index_dir = _build_index_dir(tmp_path, chunks, [V0, V1, V2])
    query(
        question="What is RAG?", index_dir=index_dir,
        embedder=_mock_embedder(V0), runtime=_mock_runtime("hi"),
        k=3, verbose=False,
    )
    captured = capsys.readouterr()
    assert captured.out.count("report.txt") == 1


def test_query_sources_shows_multiple_different_documents(tmp_path, capsys):
    chunks = [
        _chunk("About RAG.", "/data/report.txt", chunk_index=0),
        _chunk("About FAISS.", "/data/guide.md", chunk_index=0),
    ]
    index_dir = _build_index_dir(tmp_path, chunks, [V0, V1])
    query(
        question="Explain everything", index_dir=index_dir,
        embedder=_mock_embedder(V0), runtime=_mock_runtime("hi"),
        k=2, verbose=False,
    )
    captured = capsys.readouterr()
    assert "report.txt" in captured.out
    assert "guide.md" in captured.out


def test_query_result_sources_field_matches_retrieved_chunks(tmp_path):
    chunks = [_chunk("About RAG.", "/data/report.txt", chunk_index=0)]
    index_dir = _build_index_dir(tmp_path, chunks, [V0])
    result = query(
        question="What is RAG?", index_dir=index_dir,
        embedder=_mock_embedder(V0), runtime=_mock_runtime("hi"),
        verbose=False,
    )
    assert len(result.sources) == 1
    assert result.sources[0].source == "/data/report.txt"

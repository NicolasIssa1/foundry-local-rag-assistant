"""
Tests for the query pipeline — src/pipeline/query.py.

Uses a real VectorIndex + ChunkStore written to tmp_path (matching what
'python main.py index' produces on disk), with the embedder and Foundry
runtime mocked so no Foundry Local installation is required.
"""
import pytest
from pathlib import Path
from unittest.mock import MagicMock

from src.pipeline.query import query, QueryResult, NO_RELEVANT_RESULTS_MESSAGE
from src.vectorstore.index import VectorIndex
from src.vectorstore.store import ChunkStore
from src.ingestion.models import Chunk

DIM = 4
V0 = [1.0, 0.0, 0.0, 0.0]
V1 = [0.0, 1.0, 0.0, 0.0]
V2 = [0.0, 0.0, 1.0, 0.0]

# faiss.IndexFlatL2 returns SQUARED L2 distance: for orthonormal unit
# vectors this is 2.0, which exceeds DEFAULT_DISTANCE_THRESHOLD (1.25).


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
    """Uses distance_threshold=None: this test targets Sources *display*
    (dedup/formatting), not relevance filtering, so it opts out of filtering
    explicitly rather than depending on synthetic test-vector distances
    happening to clear whatever threshold filtering defaults to."""
    chunks = [
        _chunk("About RAG.", "/data/report.txt", chunk_index=0),
        _chunk("About FAISS.", "/data/guide.md", chunk_index=0),
    ]
    index_dir = _build_index_dir(tmp_path, chunks, [V0, V1])
    query(
        question="Explain everything", index_dir=index_dir,
        embedder=_mock_embedder(V0), runtime=_mock_runtime("hi"),
        k=2, distance_threshold=None, verbose=False,
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


# ── Relevance threshold — zero-result behaviour ──────────────────────────────────

def test_query_returns_deterministic_message_when_nothing_passes_threshold(tmp_path):
    """Query vector is far (orthogonal) from the only indexed chunk; a strict
    threshold must reject it and return the deterministic not-found message."""
    chunks = [_chunk("Unrelated content.", "/data/report.txt")]
    index_dir = _build_index_dir(tmp_path, chunks, [V1])
    result = query(
        question="totally unrelated question", index_dir=index_dir,
        embedder=_mock_embedder(V0), runtime=_mock_runtime("should not be used"),
        distance_threshold=0.1, verbose=False,
    )
    assert result.answer == NO_RELEVANT_RESULTS_MESSAGE


def test_query_sources_empty_when_nothing_passes_threshold(tmp_path):
    chunks = [_chunk("Unrelated content.", "/data/report.txt")]
    index_dir = _build_index_dir(tmp_path, chunks, [V1])
    result = query(
        question="totally unrelated question", index_dir=index_dir,
        embedder=_mock_embedder(V0), runtime=_mock_runtime("should not be used"),
        distance_threshold=0.1, verbose=False,
    )
    assert result.sources == []


def test_query_llm_not_called_when_nothing_passes_threshold(tmp_path):
    chunks = [_chunk("Unrelated content.", "/data/report.txt")]
    index_dir = _build_index_dir(tmp_path, chunks, [V1])
    runtime = _mock_runtime("should not be used")
    query(
        question="totally unrelated question", index_dir=index_dir,
        embedder=_mock_embedder(V0), runtime=runtime,
        distance_threshold=0.1, verbose=False,
    )
    runtime.chat.assert_not_called()
    runtime.stream_chat.assert_not_called()


def test_query_no_sources_section_when_nothing_passes_threshold(tmp_path, capsys):
    chunks = [_chunk("Unrelated content.", "/data/report.txt")]
    index_dir = _build_index_dir(tmp_path, chunks, [V1])
    query(
        question="totally unrelated question", index_dir=index_dir,
        embedder=_mock_embedder(V0), runtime=_mock_runtime("should not be used"),
        distance_threshold=0.1, verbose=False,
    )
    captured = capsys.readouterr()
    assert "Sources:" not in captured.out
    assert "report.txt" not in captured.out


def test_query_prints_the_not_found_message_when_nothing_passes_threshold(tmp_path, capsys):
    chunks = [_chunk("Unrelated content.", "/data/report.txt")]
    index_dir = _build_index_dir(tmp_path, chunks, [V1])
    query(
        question="totally unrelated question", index_dir=index_dir,
        embedder=_mock_embedder(V0), runtime=_mock_runtime("should not be used"),
        distance_threshold=0.1, verbose=False,
    )
    captured = capsys.readouterr()
    assert NO_RELEVANT_RESULTS_MESSAGE in captured.out


# ── Relevance threshold — mixed relevant/irrelevant results ──────────────────────

def test_query_keeps_only_relevant_chunks_from_a_mixed_set(tmp_path):
    chunks = [
        _chunk("Exact match content.", "/data/relevant.txt", chunk_index=0),
        _chunk("Unrelated content.", "/data/irrelevant.txt", chunk_index=0),
    ]
    index_dir = _build_index_dir(tmp_path, chunks, [V0, V1])
    result = query(
        question="query", index_dir=index_dir,
        embedder=_mock_embedder(V0), runtime=_mock_runtime("hi"),
        k=2, distance_threshold=1.0, verbose=False,
    )
    sources = {c.source for c in result.sources}
    assert sources == {"/data/relevant.txt"}


# ── Relevance threshold — overridable, and default still allows real matches ─────

def test_query_threshold_can_be_overridden_to_allow_far_matches(tmp_path):
    """Same far match rejected above must pass when threshold is loosened."""
    chunks = [_chunk("Unrelated content.", "/data/report.txt")]
    index_dir = _build_index_dir(tmp_path, chunks, [V1])
    result = query(
        question="totally unrelated question", index_dir=index_dir,
        embedder=_mock_embedder(V0), runtime=_mock_runtime("hi"),
        distance_threshold=100.0, verbose=False,
    )
    assert result.answer == "hi"
    assert len(result.sources) == 1


def test_query_default_threshold_rejects_orthogonal_match_without_explicit_arg(tmp_path):
    """DEFAULT_DISTANCE_THRESHOLD (1.25) must apply even when the caller passes
    no distance_threshold at all — orthogonal unit vectors have squared L2
    distance 2.0, which exceeds it."""
    chunks = [_chunk("Unrelated content.", "/data/report.txt")]
    index_dir = _build_index_dir(tmp_path, chunks, [V1])
    result = query(
        question="totally unrelated question", index_dir=index_dir,
        embedder=_mock_embedder(V0), runtime=_mock_runtime("should not be used"),
        verbose=False,
    )
    assert result.answer == NO_RELEVANT_RESULTS_MESSAGE


def test_query_normal_relevant_flow_unaffected_by_default_threshold(tmp_path, capsys):
    """An exact (distance 0) match must sail through the default threshold and
    produce a normal answer + Sources section, unchanged from before filtering."""
    chunks = [_chunk("About RAG.", "/data/report.txt")]
    index_dir = _build_index_dir(tmp_path, chunks, [V0])
    result = query(
        question="What is RAG?", index_dir=index_dir,
        embedder=_mock_embedder(V0), runtime=_mock_runtime("a real answer"),
        verbose=False,
    )
    captured = capsys.readouterr()
    assert result.answer == "a real answer"
    assert len(result.sources) == 1
    assert "Sources:" in captured.out
    assert "report.txt" in captured.out

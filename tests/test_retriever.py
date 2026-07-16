"""
Tests for Retriever.

Uses real VectorIndex and ChunkStore (in-process, no server needed) with
4-dimensional test vectors. The Embedder is mocked so tests run without
Foundry Local.

Test vector layout (dim=4):
  chunk 0  →  [1, 0, 0, 0]  stored in FAISS at position 0
  chunk 1  →  [0, 1, 0, 0]  stored in FAISS at position 1
  chunk 2  →  [0, 0, 1, 0]  stored in FAISS at position 2

  Query [1, 0, 0, 0] is identical to chunk 0 → must be first result.
  Query [0, 1, 0, 0] is identical to chunk 1 → must be first result.
"""
import pytest
from pathlib import Path
from unittest.mock import MagicMock

from src.retrieval.retriever import Retriever, DEFAULT_K, DEFAULT_DISTANCE_THRESHOLD
from src.vectorstore.index import VectorIndex
from src.vectorstore.store import ChunkStore
from src.ingestion.models import Chunk

DIM = 4

V0 = [1.0, 0.0, 0.0, 0.0]
V1 = [0.0, 1.0, 0.0, 0.0]
V2 = [0.0, 0.0, 1.0, 0.0]

# faiss.IndexFlatL2 returns SQUARED L2 distance. For orthonormal unit
# vectors a, b: ||a-b||^2 = ||a||^2 + ||b||^2 - 2(a.b) = 1 + 1 - 0 = 2.
ORTHOGONAL_DISTANCE = 2.0

TEXTS = ["RAG stands for Retrieval-Augmented Generation.",
         "Foundry Local runs AI models on your machine.",
         "FAISS performs fast nearest-neighbour search."]


def _mock_embedder(return_vector: list[float]) -> MagicMock:
    embedder = MagicMock()
    embedder.embed_one.return_value = return_vector
    return embedder


def _populated_retriever(tmp_path: Path, query_vector: list[float]) -> Retriever:
    """Build a Retriever with 3 known chunks, mock embedder returns query_vector."""
    index = VectorIndex(DIM)
    store = ChunkStore(tmp_path / "test.db")

    chunks = [
        Chunk(text=TEXTS[i], source=f"/data/doc{i}.txt",
              file_type="txt", page=1, chunk_index=i,
              start_char=0, end_char=len(TEXTS[i]))
        for i in range(3)
    ]
    store.add(chunks)
    index.add([V0, V1, V2])

    embedder = _mock_embedder(query_vector)
    return Retriever(embedder=embedder, index=index, store=store)


# ── Return type ───────────────────────────────────────────────────────────────

def test_retrieve_returns_list(tmp_path):
    r = _populated_retriever(tmp_path, V0)
    result = r.retrieve("What is RAG?")
    assert isinstance(result, list)


def test_retrieve_returns_chunk_objects(tmp_path):
    r = _populated_retriever(tmp_path, V0)
    result = r.retrieve("What is RAG?")
    assert all(isinstance(c, Chunk) for c in result)


# ── Ranking correctness ───────────────────────────────────────────────────────

def test_retrieve_returns_exact_match_first_for_v0(tmp_path):
    """Mock embedder returns V0 → chunk 0 must be first result."""
    r = _populated_retriever(tmp_path, V0)
    result = r.retrieve("What is RAG?")
    assert result[0].text == TEXTS[0]


def test_retrieve_returns_exact_match_first_for_v1(tmp_path):
    """Mock embedder returns V1 → chunk 1 must be first result."""
    r = _populated_retriever(tmp_path, V1)
    result = r.retrieve("Tell me about Foundry Local")
    assert result[0].text == TEXTS[1]


def test_retrieve_returns_all_chunks_with_k_equal_to_total(tmp_path):
    r = _populated_retriever(tmp_path, V0)
    result = r.retrieve("query", k=3)
    assert len(result) == 3


# ── k parameter ───────────────────────────────────────────────────────────────

def test_retrieve_respects_k_1(tmp_path):
    r = _populated_retriever(tmp_path, V0)
    result = r.retrieve("query", k=1)
    assert len(result) == 1


def test_retrieve_respects_k_2(tmp_path):
    r = _populated_retriever(tmp_path, V0)
    result = r.retrieve("query", k=2)
    assert len(result) == 2


def test_retrieve_uses_instance_default_k_when_not_specified(tmp_path):
    index = VectorIndex(DIM)
    store = ChunkStore(tmp_path / "test.db")
    vectors = [[float(i), 0.0, 0.0, 0.0] for i in range(10)]
    chunks = [
        Chunk(text=f"chunk {i}", source="f.txt", file_type="txt",
              page=1, chunk_index=i, start_char=0, end_char=10)
        for i in range(10)
    ]
    store.add(chunks)
    index.add(vectors)
    r = Retriever(_mock_embedder(V0), index, store, k=3)
    result = r.retrieve("query")
    assert len(result) == 3


def test_retrieve_k_larger_than_index_returns_all(tmp_path):
    r = _populated_retriever(tmp_path, V0)
    result = r.retrieve("query", k=100)
    assert len(result) == 3


# ── Error handling ────────────────────────────────────────────────────────────

def test_retrieve_empty_string_raises_value_error(tmp_path):
    r = _populated_retriever(tmp_path, V0)
    with pytest.raises(ValueError, match="non-empty"):
        r.retrieve("")


def test_retrieve_whitespace_only_raises_value_error(tmp_path):
    r = _populated_retriever(tmp_path, V0)
    with pytest.raises(ValueError):
        r.retrieve("   ")


# ── Empty index ───────────────────────────────────────────────────────────────

def test_retrieve_from_empty_index_returns_empty_list(tmp_path):
    index = VectorIndex(DIM)
    store = ChunkStore(tmp_path / "test.db")
    r = Retriever(_mock_embedder(V0), index, store)
    result = r.retrieve("anything")
    assert result == []


# ── Embedder interaction ──────────────────────────────────────────────────────

def test_retrieve_calls_embed_one_with_query(tmp_path):
    r = _populated_retriever(tmp_path, V0)
    r.retrieve("What is RAG?")
    r._embedder.embed_one.assert_called_once_with("What is RAG?")


def test_retrieve_chunk_metadata_is_preserved(tmp_path):
    r = _populated_retriever(tmp_path, V0)
    result = r.retrieve("query", k=1)
    chunk = result[0]
    assert chunk.source == "/data/doc0.txt"
    assert chunk.page == 1
    assert chunk.file_type == "txt"


# ── Default k constant ────────────────────────────────────────────────────────

def test_default_k_is_five():
    assert DEFAULT_K == 5


# ── Relevance threshold — no filtering by default (backward compatible) ─────────

def test_retrieve_default_threshold_is_none(tmp_path):
    r = _populated_retriever(tmp_path, V0)
    assert r._distance_threshold is None


def test_retrieve_without_threshold_keeps_far_matches(tmp_path):
    """With no threshold set, even a very distant match is still returned (old behaviour)."""
    r = _populated_retriever(tmp_path, V0)
    result = r.retrieve("query", k=3)
    assert len(result) == 3


# ── Relevance threshold — filtering behaviour ────────────────────────────────────

def _retriever_with_threshold(tmp_path: Path, query_vector: list[float], threshold: float) -> Retriever:
    index = VectorIndex(DIM)
    store = ChunkStore(tmp_path / "test.db")
    chunks = [
        Chunk(text=TEXTS[i], source=f"/data/doc{i}.txt",
              file_type="txt", page=1, chunk_index=i,
              start_char=0, end_char=len(TEXTS[i]))
        for i in range(3)
    ]
    store.add(chunks)
    index.add([V0, V1, V2])
    embedder = _mock_embedder(query_vector)
    return Retriever(embedder=embedder, index=index, store=store, distance_threshold=threshold)


def test_retrieve_keeps_result_below_threshold(tmp_path):
    """Exact match (distance 0) must pass any positive threshold."""
    r = _retriever_with_threshold(tmp_path, V0, threshold=0.5)
    result = r.retrieve("query", k=1)
    assert len(result) == 1
    assert result[0].text == TEXTS[0]


def test_retrieve_discards_result_above_threshold(tmp_path):
    """Orthogonal chunks (distance sqrt(2)) must be discarded by a strict threshold."""
    r = _retriever_with_threshold(tmp_path, V0, threshold=1.0)
    result = r.retrieve("query", k=3)
    # Only the exact match (chunk 0, distance 0) passes; V1/V2 (distance sqrt(2)) do not.
    assert len(result) == 1
    assert result[0].text == TEXTS[0]


def test_retrieve_threshold_boundary_is_inclusive(tmp_path):
    """distance == threshold must PASS (<=), not be excluded."""
    r = _retriever_with_threshold(tmp_path, V0, threshold=ORTHOGONAL_DISTANCE)
    result = r.retrieve("query", k=3)
    assert len(result) == 3


def test_retrieve_mixed_relevant_and_irrelevant_keeps_only_relevant(tmp_path):
    r = _retriever_with_threshold(tmp_path, V0, threshold=1.0)
    result = r.retrieve("query", k=3)
    sources = {c.source for c in result}
    assert sources == {"/data/doc0.txt"}


def test_retrieve_returns_empty_list_when_nothing_passes_threshold(tmp_path):
    """Threshold stricter than even the exact match's distance (0) => nothing can pass."""
    r = _retriever_with_threshold(tmp_path, V1, threshold=-0.001)
    result = r.retrieve("query", k=3)
    assert result == []


def test_retrieve_threshold_is_overridable_per_instance(tmp_path):
    """Two Retrievers over the same data, different thresholds, different results."""
    strict = _retriever_with_threshold(tmp_path, V0, threshold=0.5)
    loose = _retriever_with_threshold(tmp_path, V0, threshold=100.0)
    assert len(strict.retrieve("query", k=3)) == 1
    assert len(loose.retrieve("query", k=3)) == 3


def test_default_distance_threshold_constant_is_documented_direction():
    """DEFAULT_DISTANCE_THRESHOLD must be a positive float usable as a max L2 distance."""
    assert isinstance(DEFAULT_DISTANCE_THRESHOLD, float)
    assert DEFAULT_DISTANCE_THRESHOLD > 0

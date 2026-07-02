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

from src.retrieval.retriever import Retriever, DEFAULT_K
from src.vectorstore.index import VectorIndex
from src.vectorstore.store import ChunkStore
from src.ingestion.models import Chunk

DIM = 4

V0 = [1.0, 0.0, 0.0, 0.0]
V1 = [0.0, 1.0, 0.0, 0.0]
V2 = [0.0, 0.0, 1.0, 0.0]

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

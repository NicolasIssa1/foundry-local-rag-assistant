"""
Tests for src/pipeline/index.py — index_documents().

Regression coverage for a real bug found during manual testing: FAISS's
in-process index is always fresh (IDs restart at 0 every run), but the
persisted SQLite ChunkStore was never cleared before re-indexing, so its
chunk_ids kept growing across runs. After a second `index_documents()`
call, FAISS id 0 (this run's first chunk) would resolve against whatever
chunk happened to have SQLite id 0 (a stale chunk from an earlier run) —
silently serving the wrong text for every query. index_documents() must
call ChunkStore.clear() before writing so both ID spaces restart at 0
together on every run.
"""
from pathlib import Path
from unittest.mock import MagicMock

from src.pipeline.index import index_documents
from src.vectorstore.index import VectorIndex
from src.vectorstore.store import ChunkStore

EMBED_DIM = 4


def _mock_embedder():
    """Returns one deterministic vector per input text (index-based, so
    re-runs with different text still produce valid, distinct vectors)."""
    embedder = MagicMock()

    def _embed(texts: list[str]) -> list[list[float]]:
        return [[float(i == j) for j in range(EMBED_DIM)] for i in range(len(texts))]

    embedder.embed.side_effect = _embed
    return embedder


def _write_docs(data_dir: Path, contents: dict[str, str]) -> None:
    data_dir.mkdir(parents=True, exist_ok=True)
    for name, text in contents.items():
        (data_dir / name).write_text(text, encoding="utf-8")


# ── Happy path ────────────────────────────────────────────────────────────────

def test_index_documents_returns_chunk_count(tmp_path):
    data_dir = tmp_path / "data"
    _write_docs(data_dir, {"a.txt": "Hello world, this is a short document."})
    index_dir = tmp_path / "index"

    count = index_documents(data_dir, index_dir, _mock_embedder())

    assert count > 0


def test_index_documents_writes_faiss_and_sqlite_files(tmp_path):
    data_dir = tmp_path / "data"
    _write_docs(data_dir, {"a.txt": "Hello world, this is a short document."})
    index_dir = tmp_path / "index"

    index_documents(data_dir, index_dir, _mock_embedder())

    assert (index_dir / "faiss.index").exists()
    assert (index_dir / "chunks.db").exists()


def test_index_documents_faiss_and_store_counts_match(tmp_path):
    data_dir = tmp_path / "data"
    _write_docs(data_dir, {"a.txt": "Hello world, this is a short document."})
    index_dir = tmp_path / "index"

    count = index_documents(data_dir, index_dir, _mock_embedder())

    faiss_index = VectorIndex.load(index_dir / "faiss.index")
    store = ChunkStore(index_dir / "chunks.db")
    assert faiss_index.size == count
    assert store.count() == count
    store.close()


# ── Regression: re-indexing must not accumulate stale chunks ────────────────────

def test_reindexing_does_not_accumulate_rows_in_the_store(tmp_path):
    data_dir = tmp_path / "data"
    index_dir = tmp_path / "index"

    _write_docs(data_dir, {"a.txt": "First version of the document."})
    first_count = index_documents(data_dir, index_dir, _mock_embedder())

    _write_docs(data_dir, {"a.txt": "First version of the document."})
    second_count = index_documents(data_dir, index_dir, _mock_embedder())

    store = ChunkStore(index_dir / "chunks.db")
    assert store.count() == second_count
    assert store.count() == first_count  # same corpus indexed twice
    store.close()


def test_reindexing_with_different_content_replaces_old_chunks_entirely(tmp_path):
    """The core bug: after re-indexing with NEW text, a query must never be
    able to retrieve the OLD text — even though FAISS ids restart at 0
    every run, the store's ids must restart with them."""
    data_dir = tmp_path / "data"
    index_dir = tmp_path / "index"

    _write_docs(data_dir, {"a.txt": "OLD TEXT MARKER content here."})
    index_documents(data_dir, index_dir, _mock_embedder())

    _write_docs(data_dir, {"a.txt": "NEW TEXT MARKER content here."})
    index_documents(data_dir, index_dir, _mock_embedder())

    store = ChunkStore(index_dir / "chunks.db")
    all_chunks = store.get_all()
    all_text = " ".join(c.text for c in all_chunks)
    assert "NEW TEXT MARKER" in all_text
    assert "OLD TEXT MARKER" not in all_text
    store.close()


def test_reindexing_keeps_faiss_ids_aligned_with_store_ids(tmp_path):
    """FAISS id 0 must resolve, via ChunkStore, to a chunk from THIS run —
    never a leftover chunk from a previous run's now-stale id 0."""
    data_dir = tmp_path / "data"
    index_dir = tmp_path / "index"

    _write_docs(data_dir, {"a.txt": "OLD TEXT MARKER content here."})
    index_documents(data_dir, index_dir, _mock_embedder())

    _write_docs(data_dir, {"a.txt": "NEW TEXT MARKER content here."})
    index_documents(data_dir, index_dir, _mock_embedder())

    store = ChunkStore(index_dir / "chunks.db")
    chunk_zero = store.get([0])
    assert len(chunk_zero) == 1
    assert "NEW TEXT MARKER" in chunk_zero[0].text
    store.close()


def test_reindexing_three_times_still_stays_aligned(tmp_path):
    data_dir = tmp_path / "data"
    index_dir = tmp_path / "index"

    for marker in ["FIRST", "SECOND", "THIRD"]:
        _write_docs(data_dir, {"a.txt": f"{marker} TEXT MARKER content here."})
        index_documents(data_dir, index_dir, _mock_embedder())

    store = ChunkStore(index_dir / "chunks.db")
    all_text = " ".join(c.text for c in store.get_all())
    assert "THIRD TEXT MARKER" in all_text
    assert "FIRST TEXT MARKER" not in all_text
    assert "SECOND TEXT MARKER" not in all_text
    store.close()

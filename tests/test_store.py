"""
Tests for ChunkStore (SQLite metadata store).

All tests use pytest's tmp_path fixture for an isolated database file per
test — no shared state, no cleanup needed.
"""
import pytest
from pathlib import Path

from src.vectorstore.store import ChunkStore
from src.ingestion.models import Chunk


# ── Helpers ───────────────────────────────────────────────────────────────────

def _chunk(
    text: str = "sample text",
    source: str = "/data/doc.txt",
    file_type: str = "txt",
    page: int = 1,
    chunk_index: int = 0,
    start_char: int = 0,
    end_char: int = 11,
) -> Chunk:
    return Chunk(
        text=text,
        source=source,
        file_type=file_type,
        page=page,
        chunk_index=chunk_index,
        start_char=start_char,
        end_char=end_char,
    )


def _store(tmp_path: Path) -> ChunkStore:
    return ChunkStore(tmp_path / "test.db")


# ── Construction and lifecycle ────────────────────────────────────────────────

def test_store_creates_db_file(tmp_path):
    db_path = tmp_path / "chunks.db"
    with ChunkStore(db_path):
        pass
    assert db_path.exists()


def test_store_creates_parent_directories(tmp_path):
    db_path = tmp_path / "nested" / "dir" / "chunks.db"
    with ChunkStore(db_path):
        pass
    assert db_path.exists()


def test_context_manager_closes_connection(tmp_path):
    with _store(tmp_path) as store:
        store.add([_chunk()])
    # After __exit__, further operations should raise
    with pytest.raises(Exception):
        store.count()


# ── count() ───────────────────────────────────────────────────────────────────

def test_count_starts_at_zero(tmp_path):
    with _store(tmp_path) as store:
        assert store.count() == 0


def test_count_increases_after_add(tmp_path):
    with _store(tmp_path) as store:
        store.add([_chunk(), _chunk()])
        assert store.count() == 2


def test_count_accumulates_across_multiple_adds(tmp_path):
    with _store(tmp_path) as store:
        store.add([_chunk()])
        store.add([_chunk(), _chunk()])
        assert store.count() == 3


# ── add() — return values ─────────────────────────────────────────────────────

def test_add_returns_list_of_ids(tmp_path):
    with _store(tmp_path) as store:
        ids = store.add([_chunk(), _chunk()])
        assert isinstance(ids, list)
        assert all(isinstance(i, int) for i in ids)


def test_add_first_batch_ids_start_at_zero(tmp_path):
    with _store(tmp_path) as store:
        ids = store.add([_chunk(), _chunk(), _chunk()])
        assert ids == [0, 1, 2]


def test_add_second_batch_ids_continue_from_count(tmp_path):
    with _store(tmp_path) as store:
        store.add([_chunk(), _chunk()])        # IDs 0, 1
        ids = store.add([_chunk(), _chunk()])  # IDs 2, 3
        assert ids == [2, 3]


def test_add_empty_list_returns_empty_list(tmp_path):
    with _store(tmp_path) as store:
        assert store.add([]) == []


def test_add_empty_list_does_not_change_count(tmp_path):
    with _store(tmp_path) as store:
        store.add([_chunk()])
        store.add([])
        assert store.count() == 1


# ── add() — data integrity ────────────────────────────────────────────────────

def test_add_stores_text_correctly(tmp_path):
    text = "Retrieval-Augmented Generation is a technique."
    with _store(tmp_path) as store:
        store.add([_chunk(text=text)])
        result = store.get([0])
        assert result[0].text == text


def test_add_stores_source_correctly(tmp_path):
    with _store(tmp_path) as store:
        store.add([_chunk(source="/data/report.pdf")])
        assert store.get([0])[0].source == "/data/report.pdf"


def test_add_stores_file_type_correctly(tmp_path):
    with _store(tmp_path) as store:
        store.add([_chunk(file_type="md")])
        assert store.get([0])[0].file_type == "md"


def test_add_stores_page_correctly(tmp_path):
    with _store(tmp_path) as store:
        store.add([_chunk(page=7)])
        assert store.get([0])[0].page == 7


def test_add_stores_chunk_index_correctly(tmp_path):
    with _store(tmp_path) as store:
        store.add([_chunk(chunk_index=3)])
        assert store.get([0])[0].chunk_index == 3


def test_add_stores_start_char_correctly(tmp_path):
    with _store(tmp_path) as store:
        store.add([_chunk(start_char=100)])
        assert store.get([0])[0].start_char == 100


def test_add_stores_end_char_correctly(tmp_path):
    with _store(tmp_path) as store:
        store.add([_chunk(end_char=500)])
        assert store.get([0])[0].end_char == 500


def test_add_stores_unicode_text(tmp_path):
    text = "Bonjour le monde — résumé des résultats 中文 🎓"
    with _store(tmp_path) as store:
        store.add([_chunk(text=text)])
        assert store.get([0])[0].text == text


def test_add_timestamps_are_stored_in_metadata(tmp_path):
    with _store(tmp_path) as store:
        store.add([_chunk()])
        result = store.get([0])
        assert "indexed_at" in result[0].metadata
        assert result[0].metadata["indexed_at"]  # non-empty string


# ── get() ─────────────────────────────────────────────────────────────────────

def test_get_returns_list(tmp_path):
    with _store(tmp_path) as store:
        store.add([_chunk()])
        assert isinstance(store.get([0]), list)


def test_get_returns_chunk_objects(tmp_path):
    with _store(tmp_path) as store:
        store.add([_chunk()])
        result = store.get([0])
        assert all(isinstance(c, Chunk) for c in result)


def test_get_empty_ids_returns_empty_list(tmp_path):
    with _store(tmp_path) as store:
        assert store.get([]) == []


def test_get_nonexistent_id_returns_empty_list(tmp_path):
    with _store(tmp_path) as store:
        assert store.get([999]) == []


def test_get_preserves_requested_order(tmp_path):
    """get() must return chunks in the order IDs were requested, not storage order.
    This is critical: FAISS returns IDs ranked by similarity, so we must not re-sort.
    """
    with _store(tmp_path) as store:
        store.add([
            _chunk(text="first"),
            _chunk(text="second"),
            _chunk(text="third"),
        ])
        # Request in reverse order — similarity ranking from FAISS might do this
        result = store.get([2, 0, 1])
        assert result[0].text == "third"
        assert result[1].text == "first"
        assert result[2].text == "second"


def test_get_skips_missing_ids_silently(tmp_path):
    with _store(tmp_path) as store:
        store.add([_chunk(text="only one")])
        result = store.get([0, 99, 100])
        assert len(result) == 1
        assert result[0].text == "only one"


# ── get_all() ─────────────────────────────────────────────────────────────────

def test_get_all_returns_all_chunks_in_id_order(tmp_path):
    with _store(tmp_path) as store:
        store.add([_chunk(text="a"), _chunk(text="b"), _chunk(text="c")])
        result = store.get_all()
        assert [c.text for c in result] == ["a", "b", "c"]


def test_get_all_on_empty_store_returns_empty_list(tmp_path):
    with _store(tmp_path) as store:
        assert store.get_all() == []


# ── clear() ───────────────────────────────────────────────────────────────────

def test_clear_removes_all_chunks(tmp_path):
    with _store(tmp_path) as store:
        store.add([_chunk(), _chunk()])
        store.clear()
        assert store.count() == 0


def test_clear_resets_id_sequence(tmp_path):
    with _store(tmp_path) as store:
        store.add([_chunk(), _chunk()])
        store.clear()
        ids = store.add([_chunk()])
        assert ids == [0]


# ── FAISS synchronisation invariant ──────────────────────────────────────────

def test_ids_returned_by_add_match_get_keys(tmp_path):
    """The IDs returned by add() must be exactly the IDs usable with get()."""
    with _store(tmp_path) as store:
        chunks = [_chunk(text=f"chunk {i}") for i in range(5)]
        ids = store.add(chunks)
        retrieved = store.get(ids)
        assert len(retrieved) == 5
        for i, chunk in enumerate(retrieved):
            assert chunk.text == f"chunk {i}"

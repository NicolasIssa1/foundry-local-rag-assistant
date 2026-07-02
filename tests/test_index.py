"""
Tests for VectorIndex (FAISS IndexFlatL2 wrapper).

Uses small, known vectors so we can reason about expected distances by hand.
No mocking needed — FAISS runs in-process with no external dependencies.

Reference vectors (dim=4):
  v0 = [1, 0, 0, 0]
  v1 = [0, 1, 0, 0]
  v2 = [0, 0, 1, 0]
  v3 = [0, 0, 0, 1]

  L2 distance between any two of these = sqrt(2) ≈ 1.414
  Query [1, 0, 0, 0] is identical to v0 (distance 0) and equidistant from v1, v2, v3.
"""
import pytest
from pathlib import Path

from src.vectorstore.index import VectorIndex

DIM = 4

# Known unit vectors — easy to reason about distances
V0 = [1.0, 0.0, 0.0, 0.0]
V1 = [0.0, 1.0, 0.0, 0.0]
V2 = [0.0, 0.0, 1.0, 0.0]
V3 = [0.0, 0.0, 0.0, 1.0]


# ── Construction ──────────────────────────────────────────────────────────────

def test_index_creates_with_valid_dimension():
    idx = VectorIndex(DIM)
    assert idx.dimension == DIM


def test_index_raises_for_zero_dimension():
    with pytest.raises(ValueError):
        VectorIndex(0)


def test_index_raises_for_negative_dimension():
    with pytest.raises(ValueError):
        VectorIndex(-1)


# ── Properties ────────────────────────────────────────────────────────────────

def test_size_starts_at_zero():
    assert VectorIndex(DIM).size == 0


def test_size_increases_after_add():
    idx = VectorIndex(DIM)
    idx.add([V0, V1])
    assert idx.size == 2


def test_size_accumulates_across_multiple_adds():
    idx = VectorIndex(DIM)
    idx.add([V0, V1])
    idx.add([V2])
    assert idx.size == 3


def test_dimension_property_matches_constructor():
    idx = VectorIndex(128)
    assert idx.dimension == 128


# ── add() ─────────────────────────────────────────────────────────────────────

def test_add_empty_list_does_not_change_size():
    idx = VectorIndex(DIM)
    idx.add([])
    assert idx.size == 0


def test_add_wrong_dimension_raises_value_error():
    idx = VectorIndex(DIM)
    wrong_dim_vector = [1.0, 0.0]  # dim=2, not 4
    with pytest.raises(ValueError, match="dimension mismatch"):
        idx.add([wrong_dim_vector])


def test_add_single_vector():
    idx = VectorIndex(DIM)
    idx.add([V0])
    assert idx.size == 1


def test_add_batch_of_vectors():
    idx = VectorIndex(DIM)
    idx.add([V0, V1, V2, V3])
    assert idx.size == 4


# ── search() — correctness ────────────────────────────────────────────────────

def test_search_returns_list():
    idx = VectorIndex(DIM)
    idx.add([V0, V1, V2])
    result = idx.search(V0, k=1)
    assert isinstance(result, list)


def test_search_exact_match_is_first():
    """Querying with V0 should return ID 0 first (distance = 0)."""
    idx = VectorIndex(DIM)
    idx.add([V0, V1, V2, V3])
    result = idx.search(V0, k=1)
    assert result[0] == 0


def test_search_returns_k_results():
    idx = VectorIndex(DIM)
    idx.add([V0, V1, V2, V3])
    result = idx.search(V0, k=3)
    assert len(result) == 3


def test_search_results_are_integers():
    idx = VectorIndex(DIM)
    idx.add([V0, V1])
    result = idx.search(V0, k=2)
    assert all(isinstance(r, int) for r in result)


def test_search_results_are_ordered_closest_first():
    """
    V0 query: distance to V0=0 (closest), distance to V1=V2=V3=sqrt(2).
    ID 0 must come before IDs 1, 2, 3.
    """
    idx = VectorIndex(DIM)
    idx.add([V0, V1, V2, V3])
    result = idx.search(V0, k=4)
    assert result[0] == 0


def test_search_all_four_vectors_returned():
    idx = VectorIndex(DIM)
    idx.add([V0, V1, V2, V3])
    result = idx.search(V0, k=4)
    assert set(result) == {0, 1, 2, 3}


def test_search_k_larger_than_size_returns_all():
    """If k > index size, return all stored IDs without error."""
    idx = VectorIndex(DIM)
    idx.add([V0, V1])
    result = idx.search(V0, k=100)
    assert len(result) == 2


def test_search_no_minus_one_sentinels_in_result():
    """FAISS pads short results with -1; our wrapper must filter them out."""
    idx = VectorIndex(DIM)
    idx.add([V0])
    result = idx.search(V0, k=5)
    assert -1 not in result


def test_search_known_nearest_neighbour():
    """
    Add V0=[1,0,0,0] and V1=[0,1,0,0].
    Query with [0.9, 0.1, 0, 0] — closer to V0.
    """
    idx = VectorIndex(DIM)
    idx.add([V0, V1])
    query = [0.9, 0.1, 0.0, 0.0]
    result = idx.search(query, k=1)
    assert result[0] == 0   # V0 is the nearest neighbour


def test_search_second_nearest_neighbour():
    """
    Add V0, V1, V2. Query close to V1.
    V1 should be first (distance ~ 0), V0 and V2 should also appear.
    """
    idx = VectorIndex(DIM)
    idx.add([V0, V1, V2])
    query = [0.05, 0.95, 0.0, 0.0]   # almost V1
    result = idx.search(query, k=3)
    assert result[0] == 1


# ── search() — error handling ─────────────────────────────────────────────────

def test_search_empty_index_returns_empty_list():
    idx = VectorIndex(DIM)
    result = idx.search(V0, k=5)
    assert result == []


def test_search_empty_query_raises_value_error():
    idx = VectorIndex(DIM)
    idx.add([V0])
    with pytest.raises(ValueError, match="non-empty"):
        idx.search([], k=1)


def test_search_k_zero_raises_value_error():
    idx = VectorIndex(DIM)
    idx.add([V0])
    with pytest.raises(ValueError):
        idx.search(V0, k=0)


# ── Persistence ───────────────────────────────────────────────────────────────

def test_save_creates_file(tmp_path):
    idx = VectorIndex(DIM)
    idx.add([V0, V1])
    path = tmp_path / "test.index"
    idx.save(path)
    assert path.exists()


def test_save_creates_parent_directories(tmp_path):
    idx = VectorIndex(DIM)
    idx.add([V0])
    path = tmp_path / "nested" / "dir" / "test.index"
    idx.save(path)
    assert path.exists()


def test_load_restores_size(tmp_path):
    idx = VectorIndex(DIM)
    idx.add([V0, V1, V2])
    path = tmp_path / "test.index"
    idx.save(path)

    loaded = VectorIndex.load(path)
    assert loaded.size == 3


def test_load_restores_dimension(tmp_path):
    idx = VectorIndex(DIM)
    idx.add([V0])
    path = tmp_path / "test.index"
    idx.save(path)

    loaded = VectorIndex.load(path)
    assert loaded.dimension == DIM


def test_load_restores_search_results(tmp_path):
    """Search results must be identical before and after save/load."""
    idx = VectorIndex(DIM)
    idx.add([V0, V1, V2, V3])
    path = tmp_path / "test.index"
    idx.save(path)

    loaded = VectorIndex.load(path)
    result = loaded.search(V0, k=1)
    assert result[0] == 0


def test_load_nonexistent_file_raises_file_not_found(tmp_path):
    with pytest.raises(FileNotFoundError):
        VectorIndex.load(tmp_path / "missing.index")


# ── FAISS–SQLite synchronisation invariant ────────────────────────────────────

def test_sequential_id_assignment_matches_insertion_order():
    """
    IDs assigned by FAISS match insertion order.
    This is the invariant that keeps FAISS and SQLite in sync.
    """
    idx = VectorIndex(DIM)
    idx.add([V0, V1, V2, V3])
    # Each vector's closest match to itself must be its own insertion index
    for expected_id, vec in enumerate([V0, V1, V2, V3]):
        result = idx.search(vec, k=1)
        assert result[0] == expected_id, (
            f"Vector at insertion index {expected_id} was not returned as "
            f"its own nearest neighbour (got {result[0]})"
        )

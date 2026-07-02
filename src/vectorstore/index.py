from __future__ import annotations

from pathlib import Path

import faiss
import numpy as np


class VectorIndex:
    """FAISS IndexFlatL2 wrapper for exact nearest-neighbour search.

    Vectors are assigned sequential integer IDs (0, 1, 2...) in the order
    they are added — matching the chunk_id values in ChunkStore so the two
    systems stay in sync without any extra bookkeeping.

    This class deliberately hides FAISS and NumPy from callers: all inputs
    and outputs use plain Python lists. Swapping FAISS for another backend
    only requires changing this file.
    """

    def __init__(self, dimension: int) -> None:
        if dimension <= 0:
            raise ValueError(f"dimension must be positive, got {dimension}")
        self._index = faiss.IndexFlatL2(dimension)

    # ── Properties ────────────────────────────────────────────────────────────

    @property
    def size(self) -> int:
        """Number of vectors currently stored in the index."""
        return int(self._index.ntotal)

    @property
    def dimension(self) -> int:
        """Dimensionality of vectors this index was built for."""
        return int(self._index.d)

    # ── Write ─────────────────────────────────────────────────────────────────

    def add(self, vectors: list[list[float]]) -> None:
        """Add vectors to the index.

        Each vector must have exactly self.dimension elements.
        FAISS assigns IDs sequentially from the current size, so the first
        call assigns 0..len(vectors)-1, the next call continues from there.
        """
        if not vectors:
            return

        arr = np.array(vectors, dtype=np.float32)

        if arr.ndim != 2:
            raise ValueError(
                f"Expected a 2-D array of shape (n, {self.dimension}), "
                f"got shape {arr.shape}"
            )
        if arr.shape[1] != self.dimension:
            raise ValueError(
                f"Vector dimension mismatch: index expects {self.dimension}, "
                f"got {arr.shape[1]}"
            )

        self._index.add(arr)

    # ── Read ──────────────────────────────────────────────────────────────────

    def search(self, query_vector: list[float], k: int = 5) -> list[int]:
        """Return the IDs of the k nearest vectors to query_vector.

        Results are ordered closest-first. If the index holds fewer than k
        vectors, all stored IDs are returned. FAISS -1 sentinel values
        (padding for under-full results) are filtered out.
        """
        if not query_vector:
            raise ValueError("query_vector must be a non-empty list")
        if k <= 0:
            raise ValueError(f"k must be positive, got {k}")
        if self.size == 0:
            return []

        k_actual = min(k, self.size)
        query = np.array([query_vector], dtype=np.float32)
        _, indices = self._index.search(query, k_actual)

        return [int(i) for i in indices[0] if i != -1]

    # ── Persistence ───────────────────────────────────────────────────────────

    def save(self, path: str | Path) -> None:
        """Write the index to disk. Creates parent directories if needed."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        faiss.write_index(self._index, str(path))

    @classmethod
    def load(cls, path: str | Path) -> VectorIndex:
        """Load an index from disk without needing to know the dimension upfront."""
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Index file not found: {path}")
        instance = cls.__new__(cls)
        instance._index = faiss.read_index(str(path))
        return instance

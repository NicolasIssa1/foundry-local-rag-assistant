from __future__ import annotations

from ..embeddings.embedder import Embedder
from ..ingestion.models import Chunk
from ..vectorstore.index import VectorIndex
from ..vectorstore.store import ChunkStore

DEFAULT_K = 5


class Retriever:
    """Orchestrates the query-time pipeline.

    Embeds a natural-language question, finds the nearest chunk vectors in
    FAISS, and fetches the corresponding text from SQLite. Returns chunks
    ordered from most to least relevant.
    """

    def __init__(
        self,
        embedder: Embedder,
        index: VectorIndex,
        store: ChunkStore,
        k: int = DEFAULT_K,
    ) -> None:
        self._embedder = embedder
        self._index = index
        self._store = store
        self._k = k

    def retrieve(self, query: str, k: int | None = None) -> list[Chunk]:
        """Return the top-k most relevant chunks for the given query.

        Args:
            query: The user's natural-language question.
            k:     Number of chunks to return. Falls back to the instance
                   default when omitted.

        Returns:
            Chunks ordered closest-first (index 0 is the best match).
        """
        if not query or not query.strip():
            raise ValueError("query must be a non-empty string")

        actual_k = k if k is not None else self._k

        query_vector = self._embedder.embed_one(query)
        chunk_ids = self._index.search(query_vector, k=actual_k)
        return self._store.get(chunk_ids)

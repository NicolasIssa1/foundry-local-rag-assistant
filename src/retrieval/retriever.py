from __future__ import annotations

from ..embeddings.embedder import Embedder
from ..ingestion.models import Chunk
from ..vectorstore.index import VectorIndex
from ..vectorstore.store import ChunkStore

DEFAULT_K = 5

# VectorIndex is backed by faiss.IndexFlatL2, which returns SQUARED L2
# (Euclidean) distance, not similarity: LOWER means MORE similar (0 =
# identical). This threshold is therefore a maximum — chunks are kept when
# distance <= threshold, discarded when distance > threshold. Do not confuse
# this with a cosine-similarity/inner-product score, where the comparison
# direction is reversed (higher = better).
#
# Empirically derived (2026-07-16) against this project's sample corpus
# using qwen3-embedding-0.6b: on-topic questions' best-match squared L2
# distances fell in 0.60-1.22; off-topic questions' best-match distances
# never beat 1.38. 1.25 sits in that gap, biased toward the off-topic side
# so genuine matches aren't over-filtered. Revisit if the corpus or
# embedding model changes.
DEFAULT_DISTANCE_THRESHOLD = 1.25


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
        distance_threshold: float | None = None,
    ) -> None:
        self._embedder = embedder
        self._index = index
        self._store = store
        self._k = k
        self._distance_threshold = distance_threshold

    def retrieve(self, query: str, k: int | None = None) -> list[Chunk]:
        """Return the top-k most relevant chunks for the given query.

        Args:
            query: The user's natural-language question.
            k:     Number of chunks to return. Falls back to the instance
                   default when omitted.

        Chunks whose squared L2 distance exceeds ``distance_threshold`` (if
        set) are discarded — top-k is still computed first, then filtered,
        so k never grows results, it only ever shrinks them. With no
        threshold set (the default), behaviour is unchanged from before
        filtering existed.

        Returns:
            Chunks ordered closest-first (index 0 is the best match). May be
            an empty list if nothing passes the threshold.
        """
        if not query or not query.strip():
            raise ValueError("query must be a non-empty string")

        actual_k = k if k is not None else self._k

        query_vector = self._embedder.embed_one(query)
        results = self._index.search_with_scores(query_vector, k=actual_k)

        if self._distance_threshold is not None:
            results = [
                (chunk_id, distance)
                for chunk_id, distance in results
                if distance <= self._distance_threshold
            ]

        chunk_ids = [chunk_id for chunk_id, _ in results]
        return self._store.get(chunk_ids)

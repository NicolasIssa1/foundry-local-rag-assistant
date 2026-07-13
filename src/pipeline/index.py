from __future__ import annotations

from pathlib import Path

from ..embeddings.foundry_embedder import FoundryEmbedder
from ..ingestion import chunk, clean, load
from ..vectorstore.index import VectorIndex
from ..vectorstore.store import ChunkStore

DEFAULT_CHUNK_SIZE = 1000
DEFAULT_OVERLAP = 200
_INDEX_FILE = "faiss.index"
_DB_FILE = "chunks.db"


def index_documents(
    data_dir: str | Path,
    index_dir: str | Path,
    embedder: FoundryEmbedder,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    overlap: int = DEFAULT_OVERLAP,
    verbose: bool = True,
) -> int:
    """Load, clean, chunk, embed, and store all documents in data_dir.

    Writes two files to index_dir:
      faiss.index  — the FAISS vector index
      chunks.db    — the SQLite metadata store

    If index_dir already contains a previous index it is overwritten.
    Returns the total number of chunks indexed.
    """
    data_dir = Path(data_dir)
    index_dir = Path(index_dir)
    index_dir.mkdir(parents=True, exist_ok=True)

    def _log(msg: str) -> None:
        if verbose:
            print(msg)

    # ── Stage 1: Load ────────────────────────────────────────────────────────
    _log(f"[index] Loading documents from {data_dir} ...")
    documents = load(data_dir)
    if not documents:
        raise ValueError(f"No supported documents found in {data_dir}")
    _log(f"[index] Loaded {len(documents)} document(s).")

    # ── Stage 2: Clean ───────────────────────────────────────────────────────
    cleaned = [clean(doc) for doc in documents]

    # ── Stage 3: Chunk ───────────────────────────────────────────────────────
    all_chunks = []
    for doc in cleaned:
        all_chunks.extend(chunk(doc, chunk_size=chunk_size, overlap=overlap))
    _log(f"[index] Produced {len(all_chunks)} chunk(s).")

    if not all_chunks:
        raise ValueError("All documents were empty after cleaning.")

    # ── Stage 4: Embed ───────────────────────────────────────────────────────
    _log("[index] Embedding chunks (this may take a moment) ...")
    texts = [c.text for c in all_chunks]
    vectors = embedder.embed(texts)

    # Detect dimension from the first vector returned by the model.
    dimension = len(vectors[0])
    _log(f"[index] Embedding dimension: {dimension}")

    # ── Stage 5: Store ───────────────────────────────────────────────────────
    store = ChunkStore(index_dir / _DB_FILE)
    faiss_index = VectorIndex(dimension=dimension)

    store.add(all_chunks)
    faiss_index.add(vectors)

    faiss_index.save(index_dir / _INDEX_FILE)
    store.close()

    _log(f"[index] Saved index to {index_dir}")
    return len(all_chunks)

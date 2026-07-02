"""
M4 pipeline demonstration — runs without Foundry Local.

Shows the complete flow:
  data/ files → load → clean → chunk → (mock embed) → FAISS → SQLite → retrieve → prompt

Since embedding requires Foundry Local, this demo uses random vectors so
the pipeline mechanics are visible even offline. Real semantic retrieval
will work once the LLM client (M5) is wired in with Foundry Local running.

Run from the project root:
    python scripts/demo_m4.py
"""
from __future__ import annotations

import random
import sys
import tempfile
from pathlib import Path

# Ensure the project root is on the path regardless of working directory
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.ingestion import load, clean, chunk
from src.prompt import build
from src.retrieval import Retriever
from src.vectorstore import ChunkStore, VectorIndex

# ── Configuration ─────────────────────────────────────────────────────────────

DATA_DIR = Path("data")
CHUNK_SIZE = 500
OVERLAP = 100
TOP_K = 3
EMBEDDING_DIM = 16      # small fake dimension for the demo
DEMO_QUESTION = "What is Retrieval-Augmented Generation and why is it useful?"

DIVIDER = "─" * 70


def _section(title: str) -> None:
    print(f"\n{DIVIDER}")
    print(f"  {title}")
    print(DIVIDER)


def _fake_embed(texts: list[str], dim: int) -> list[list[float]]:
    """Deterministic fake embeddings — same text always gets the same vector."""
    vectors = []
    for text in texts:
        seed = sum(ord(c) for c in text[:50])
        rng = random.Random(seed)
        vectors.append([rng.uniform(-1, 1) for _ in range(dim)])
    return vectors


def main() -> None:
    print("\nFoundry Local RAG Assistant — M4 Pipeline Demo")
    print("(Using mock embeddings — no Foundry Local required)\n")

    # ── Stage 1: Ingestion ────────────────────────────────────────────────────
    _section("STAGE 1 — Load documents")
    documents = load(DATA_DIR)
    print(f"  Loaded {len(documents)} document(s):")
    for doc in documents:
        print(f"    • {Path(doc.source).name}  ({len(doc.content)} chars)")

    _section("STAGE 2 — Clean documents")
    cleaned_docs = [clean(doc) for doc in documents]
    for doc in cleaned_docs:
        print(f"  {Path(doc.source).name}: {len(doc.content)} chars after cleaning")

    _section("STAGE 3 — Chunk documents")
    all_chunks = []
    for doc in cleaned_docs:
        doc_chunks = chunk(doc, chunk_size=CHUNK_SIZE, overlap=OVERLAP)
        all_chunks.extend(doc_chunks)
        print(f"  {Path(doc.source).name}: {len(doc_chunks)} chunk(s)")
    print(f"\n  Total chunks: {len(all_chunks)}")

    if not all_chunks:
        print("\n  No chunks produced — check that data/ contains .txt or .md files.")
        return

    # Print the first chunk as an example
    print(f"\n  Example — Chunk 0 (first {120} chars):")
    print(f"  └─ {all_chunks[0].text[:120].replace(chr(10), ' ')}...")

    # ── Stage 2: Build index ──────────────────────────────────────────────────
    _section("STAGE 4 — Embed and index chunks (mock embeddings)")
    texts = [c.text for c in all_chunks]
    vectors = _fake_embed(texts, dim=EMBEDDING_DIM)
    print(f"  Generated {len(vectors)} mock vectors (dim={EMBEDDING_DIM})")

    with tempfile.TemporaryDirectory() as tmpdir:
        store = ChunkStore(Path(tmpdir) / "chunks.db")
        index = VectorIndex(dimension=EMBEDDING_DIM)

        assigned_ids = store.add(all_chunks)
        index.add(vectors)

        print(f"  SQLite: {store.count()} rows stored")
        print(f"  FAISS:  {index.size} vectors stored")
        assert len(assigned_ids) == index.size, "ID mismatch — synchronisation broken"
        print("  ✓ FAISS and SQLite ID counts match")

        # ── Stage 3: Retrieve ─────────────────────────────────────────────────
        _section("STAGE 5 — Retrieve top-k chunks for demo question")
        print(f"  Question: \"{DEMO_QUESTION}\"\n")

        # Mock embedder: embed the query with the same fake function
        from unittest.mock import MagicMock
        mock_embedder = MagicMock()
        mock_embedder.embed_one.return_value = _fake_embed([DEMO_QUESTION], EMBEDDING_DIM)[0]

        retriever = Retriever(embedder=mock_embedder, index=index, store=store, k=TOP_K)
        retrieved = retriever.retrieve(DEMO_QUESTION)

        print(f"  Retrieved {len(retrieved)} chunk(s) (k={TOP_K}):\n")
        for rank, c in enumerate(retrieved, start=1):
            print(f"  [{rank}] {Path(c.source).name}, page {c.page}")
            print(f"       {c.text[:100].replace(chr(10), ' ')}...")
            print()

        # ── Stage 4: Build prompt ─────────────────────────────────────────────
        _section("STAGE 6 — Build final LLM prompt")
        prompt = build(retrieved, DEMO_QUESTION)

        print()
        print(prompt)
        print()

        store.close()

    _section("SUMMARY")
    print(f"  Documents loaded   : {len(documents)}")
    print(f"  Total chunks       : {len(all_chunks)}")
    print(f"  Vectors indexed    : {len(vectors)}")
    print(f"  Chunks retrieved   : {len(retrieved)}")
    print(f"  Prompt length      : {len(prompt)} characters")
    print()
    print("  All pipeline stages verified. Ready for M5: LLM integration.")
    print()


if __name__ == "__main__":
    main()

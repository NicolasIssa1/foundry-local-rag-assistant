"""
M5 end-to-end demo — live Foundry Local models required.

Runs the complete RAG pipeline with real embeddings and a real chat model:
  1. Initialise the Foundry Local SDK and load both models.
  2. Load, clean, and chunk the sample documents.
  3. Embed every chunk with qwen3-embedding-0.6b (native, in-process).
  4. Store vectors in FAISS and metadata in SQLite.
  5. Embed the demo question and retrieve the top-k most relevant chunks.
  6. Build a RAG prompt with numbered source citations.
  7. Stream the answer from qwen2.5-0.5b token by token.

Run from the project root:
    python scripts/demo_m5.py

The index is saved to data/index/ so 'python main.py query ...' works
immediately after this script completes.
"""
from __future__ import annotations

import sys
import tempfile
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.embeddings.foundry_embedder import FoundryEmbedder
from src.ingestion import chunk, clean, load
from src.llm.client import DEFAULT_CHAT_ALIAS, DEFAULT_EMBED_ALIAS, FoundryRuntime
from src.pipeline.index import index_documents
from src.pipeline.query import query
from src.prompt.builder import build
from src.retrieval.retriever import Retriever
from src.vectorstore.index import VectorIndex
from src.vectorstore.store import ChunkStore

DATA_DIR = Path("data")
INDEX_DIR = Path("data/index")
DEMO_QUESTIONS = [
    "What is Retrieval-Augmented Generation and why is it useful?",
    "How does Foundry Local serve models locally?",
]
TOP_K = 3
DIVIDER = "─" * 70


def _section(title: str) -> None:
    print(f"\n{DIVIDER}")
    print(f"  {title}")
    print(DIVIDER)


def main() -> None:
    print("\nFoundry Local RAG Assistant — M5 Live Demo")
    print("Using real embedding and chat models (Foundry Local required)\n")

    # ── Stage 1: Initialise SDK and load models ───────────────────────────────
    _section("STAGE 1 — Initialise Foundry Local SDK")
    t0 = time.perf_counter()
    runtime = FoundryRuntime(
        embed_alias=DEFAULT_EMBED_ALIAS,
        chat_alias=DEFAULT_CHAT_ALIAS,
        verbose=True,
    )
    t1 = time.perf_counter()
    print(f"\n  Models ready in {t1 - t0:.1f}s")
    print(f"  Embedding model : {DEFAULT_EMBED_ALIAS}")
    print(f"  Chat model      : {DEFAULT_CHAT_ALIAS}")

    embedder = FoundryEmbedder(
        client=runtime.get_embedding_client(),
        model=DEFAULT_EMBED_ALIAS,
    )

    # ── Stage 2: Index documents ──────────────────────────────────────────────
    _section("STAGE 2 — Index documents")
    t0 = time.perf_counter()
    count = index_documents(
        data_dir=DATA_DIR,
        index_dir=INDEX_DIR,
        embedder=embedder,
        verbose=True,
    )
    t1 = time.perf_counter()
    print(f"\n  {count} chunk(s) indexed in {t1 - t0:.1f}s")
    print(f"  Index saved to: {INDEX_DIR}/")

    # ── Stage 3: Query ────────────────────────────────────────────────────────
    for question in DEMO_QUESTIONS:
        _section(f"STAGE 3 — Query")
        print(f"  Question: {question!r}\n")

        t0 = time.perf_counter()
        print("  Answer (streaming):\n")
        answer = query(
            question=question,
            index_dir=INDEX_DIR,
            embedder=embedder,
            runtime=runtime,
            k=TOP_K,
            stream=True,
            verbose=False,
        )
        t1 = time.perf_counter()
        print(f"\n\n  Answer generated in {t1 - t0:.1f}s")

    # ── Summary ───────────────────────────────────────────────────────────────
    _section("SUMMARY")
    print(f"  Documents indexed : {DATA_DIR}")
    print(f"  Chunks stored     : {count}")
    print(f"  Embedding model   : {DEFAULT_EMBED_ALIAS}")
    print(f"  Chat model        : {DEFAULT_CHAT_ALIAS}")
    print(f"  Questions asked   : {len(DEMO_QUESTIONS)}")
    print()
    print("  To ask your own questions:")
    print(f"    python main.py query \"<your question>\"")
    print()

    runtime.close()


if __name__ == "__main__":
    main()

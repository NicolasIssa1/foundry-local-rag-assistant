from __future__ import annotations

from pathlib import Path
from typing import Callable, Iterator, NamedTuple

from ..embeddings.foundry_embedder import FoundryEmbedder
from ..ingestion.models import Chunk
from ..llm.client import FoundryRuntime
from ..prompt.builder import build, format_sources
from ..retrieval.retriever import Retriever
from ..vectorstore.index import VectorIndex
from ..vectorstore.store import ChunkStore

_INDEX_FILE = "faiss.index"
_DB_FILE = "chunks.db"
DEFAULT_K = 5


class QueryResult(NamedTuple):
    """The generated answer paired with the chunks it was grounded in."""

    answer: str
    sources: list[Chunk]


def query(
    question: str,
    index_dir: str | Path,
    embedder: FoundryEmbedder,
    runtime: FoundryRuntime,
    k: int = DEFAULT_K,
    stream: bool = True,
    verbose: bool = True,
) -> QueryResult:
    """Answer a question using the pre-built index in index_dir.

    Steps:
      1. Load FAISS index and SQLite store from index_dir.
      2. Embed the question with the provided embedder.
      3. Retrieve the top-k most relevant chunks.
      4. Build a RAG prompt with source citations.
      5. Call the chat model (streaming by default) and print the answer.
      6. Print a Sources section built directly from the retrieved chunks
         (never from the model's own output).

    Returns a QueryResult(answer, sources).
    """
    index_dir = Path(index_dir)
    index_path = index_dir / _INDEX_FILE
    db_path = index_dir / _DB_FILE

    if not index_path.exists() or not db_path.exists():
        raise FileNotFoundError(
            f"No index found in {index_dir}. Run 'python main.py index' first."
        )

    def _log(msg: str) -> None:
        if verbose:
            print(msg)

    # ── Load index ───────────────────────────────────────────────────────────
    faiss_index = VectorIndex.load(index_path)
    store = ChunkStore(db_path)

    # ── Retrieve ─────────────────────────────────────────────────────────────
    retriever = Retriever(embedder=embedder, index=faiss_index, store=store, k=k)
    chunks = retriever.retrieve(question)
    _log(f"[query] Retrieved {len(chunks)} chunk(s) for: {question!r}")

    # ── Build prompt ─────────────────────────────────────────────────────────
    prompt = build(chunks, question)

    # ── Call LLM ─────────────────────────────────────────────────────────────
    messages = [{"role": "user", "content": prompt}]
    answer_parts: list[str] = []

    if stream:
        for token in runtime.stream_chat(messages):
            print(token, end="", flush=True)
            answer_parts.append(token)
        print()
    else:
        answer = runtime.chat(messages)
        answer_parts.append(answer)

    # ── Sources ──────────────────────────────────────────────────────────────
    # Always shown, regardless of verbose — built from the retrieved chunks
    # themselves, never from model output.
    print()
    print(format_sources(chunks))

    store.close()
    return QueryResult(answer="".join(answer_parts), sources=chunks)

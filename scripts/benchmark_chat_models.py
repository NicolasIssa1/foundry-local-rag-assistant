"""
TEMPORARY diagnostic script — chat-model benchmark (M6).

Compares chat-model candidates against the existing indexed documents using
IDENTICAL retrieval results and prompts for every model. The embedding model,
FAISS index, and relevance threshold are all left unchanged.

Not part of the application; not intended to be committed or kept long-term.

Run from the project root:
    python scripts/benchmark_chat_models.py
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from foundry_local_sdk import Configuration, FoundryLocalManager

from src.embeddings.foundry_embedder import FoundryEmbedder
from src.prompt.builder import build
from src.retrieval.retriever import DEFAULT_DISTANCE_THRESHOLD, Retriever
from src.vectorstore.index import VectorIndex
from src.vectorstore.store import ChunkStore

INDEX_DIR = Path("data/index")
EMBED_ALIAS = "qwen3-embedding-0.6b"
BASELINE_ALIAS = "qwen2.5-0.5b"
CANDIDATE_ALIASES = ["qwen3-1.7b"]  # one at a time, per instructions

ANSWERABLE_QUESTIONS = [
    "What is retrieval-augmented generation?",
    "How does semantic search work?",
    "How does Foundry Local serve models locally?",
]
UNRELATED_QUESTIONS = [
    "Who won the 2026 FIFA World Cup?",
    "What is the weather in Beirut today?",
]

DIVIDER = "=" * 78


def section(title: str) -> None:
    print(f"\n{DIVIDER}\n{title}\n{DIVIDER}")


# ── SDK init (same app_name as production => same on-disk cache) ────────────
config = Configuration(app_name="foundry_local_rag_assistant")
FoundryLocalManager.initialize(config)
manager = FoundryLocalManager.instance

section("STAGE 0 — Execution providers")
t0 = time.perf_counter()
manager.download_and_register_eps()
print(f"EPs ready in {time.perf_counter() - t0:.1f}s")

# ── Embedding model (loaded ONCE, reused for every chat-model candidate) ────
section("STAGE 1 — Embedding model (fixed across all candidates)")
embed_model = manager.catalog.get_model(EMBED_ALIAS)
if embed_model is None:
    raise SystemExit(f"Embedding alias not found in catalog: {EMBED_ALIAS!r}")
print(f"alias={EMBED_ALIAS}  resolved_id={embed_model.id}  is_cached={embed_model.is_cached}")
t0 = time.perf_counter()
embed_model.download()
t1 = time.perf_counter()
embed_model.load()
t2 = time.perf_counter()
print(f"embedding download={t1-t0:.2f}s  load={t2-t1:.2f}s")
embedder = FoundryEmbedder(client=embed_model.get_embedding_client(), model=EMBED_ALIAS)

# ── Retrieval (computed ONCE per question, reused for every chat model) ─────
section("STAGE 2 — Retrieval (fixed index + threshold, computed once)")
faiss_index = VectorIndex.load(INDEX_DIR / "faiss.index")
store = ChunkStore(INDEX_DIR / "chunks.db")
retriever = Retriever(
    embedder=embedder, index=faiss_index, store=store,
    distance_threshold=DEFAULT_DISTANCE_THRESHOLD,
)
print(f"distance_threshold={DEFAULT_DISTANCE_THRESHOLD} (unchanged)")

precomputed = {}  # question -> {"chunks": [...], "prompt": str}
for q in ANSWERABLE_QUESTIONS + UNRELATED_QUESTIONS:
    chunks = retriever.retrieve(q)
    prompt = build(chunks, q) if chunks else None
    precomputed[q] = {"chunks": chunks, "prompt": prompt}
    kind = "ANSWERABLE" if q in ANSWERABLE_QUESTIONS else "UNRELATED"
    sources = sorted({Path(c.source).name for c in chunks})
    print(f"[{kind}] {q!r} -> {len(chunks)} chunk(s) passed threshold; sources={sources}")

print("\nGuard check — unrelated questions must be blocked BEFORE any LLM call:")
for q in UNRELATED_QUESTIONS:
    blocked = len(precomputed[q]["chunks"]) == 0
    print(f"  {q!r}: blocked_by_retrieval={blocked}")


def benchmark_model(alias: str, is_baseline: bool) -> None:
    section(f"STAGE 3 — Chat model: {alias} {'(BASELINE)' if is_baseline else '(CANDIDATE)'}")
    model = manager.catalog.get_model(alias)
    if model is None:
        print(f"ERROR: alias not found in catalog: {alias!r}. Skipping.")
        return

    print(f"alias={alias}  resolved_id={model.id}  file_size_mb={model.info.file_size_mb}  "
          f"context_length={model.context_length}  capabilities={model.capabilities}  "
          f"is_cached_before={model.is_cached}")

    was_cached = model.is_cached
    try:
        t0 = time.perf_counter()
        if not was_cached:
            model.download()
        t1 = time.perf_counter()
        download_time = (t1 - t0) if not was_cached else 0.0

        model.load()
        t2 = time.perf_counter()
        load_time = t2 - t1

        print(f"download_time={'N/A (already cached)' if was_cached else f'{download_time:.2f}s'}  "
              f"load_time={load_time:.2f}s")

        chat_client = model.get_chat_client()

        for q in ANSWERABLE_QUESTIONS:
            prompt = precomputed[q]["prompt"]
            messages = [{"role": "user", "content": prompt}]

            print(f"\n--- Question: {q!r} ---")
            t_start = time.perf_counter()
            first_token_time = None
            answer_parts = []
            for chunk in chat_client.complete_streaming_chat(messages):
                if chunk.choices and chunk.choices[0].delta.content:
                    if first_token_time is None:
                        first_token_time = time.perf_counter()
                    answer_parts.append(chunk.choices[0].delta.content)
            t_end = time.perf_counter()

            answer = "".join(answer_parts)
            ttft = (first_token_time - t_start) if first_token_time else None
            total = t_end - t_start

            print(f"time_to_first_token={f'{ttft:.2f}s' if ttft is not None else 'N/A'}  "
                  f"total_generation_time={total:.2f}s")
            print(f"answer:\n{answer}\n")

    except Exception as e:
        print(f"ERROR benchmarking {alias!r}: {type(e).__name__}: {e}")
    finally:
        try:
            model.unload()
            print(f"[{alias}] unloaded cleanly.")
        except Exception as e:
            print(f"WARNING: failed to unload {alias!r}: {type(e).__name__}: {e}")


benchmark_model(BASELINE_ALIAS, is_baseline=True)
for alias in CANDIDATE_ALIASES:
    benchmark_model(alias, is_baseline=False)

store.close()
embed_model.unload()
section("DONE")

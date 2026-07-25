"""
M6 RAG evaluation script.

Runs the reusable evaluation dataset (src/evaluation/eval_dataset.json)
against the real, currently-built index using the production embedding
model, retriever, relevance threshold, prompt builder, and chat model —
unchanged from the application. Not part of the application; nothing in
main.py imports this. Writes nothing to disk — report goes to stdout only.

Run from the project root (after `python main.py index`):
    python scripts/evaluate_rag.py
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.embeddings.foundry_embedder import FoundryEmbedder
from src.evaluation.dataset import load_dataset
from src.evaluation.runner import run_dataset
from src.evaluation.scoring import summarize
from src.llm.client import FoundryRuntime
from src.retrieval.retriever import DEFAULT_DISTANCE_THRESHOLD, Retriever
from src.vectorstore.index import VectorIndex
from src.vectorstore.store import ChunkStore

INDEX_DIR = Path(__file__).parent.parent / "data" / "index"
DIVIDER = "=" * 90


def main() -> int:
    if not (INDEX_DIR / "faiss.index").exists() or not (INDEX_DIR / "chunks.db").exists():
        print(f"No index found in {INDEX_DIR}. Run 'python main.py index' first.", file=sys.stderr)
        return 1

    cases = load_dataset()
    print(f"Loaded {len(cases)} evaluation case(s) from src/evaluation/eval_dataset.json")
    print(f"distance_threshold={DEFAULT_DISTANCE_THRESHOLD} (unchanged from the application)")

    with FoundryRuntime() as runtime:
        embedder = FoundryEmbedder(
            client=runtime.get_embedding_client(), model=runtime._embed_alias
        )
        faiss_index = VectorIndex.load(INDEX_DIR / "faiss.index")
        store = ChunkStore(INDEX_DIR / "chunks.db")
        retriever = Retriever(
            embedder=embedder, index=faiss_index, store=store,
            distance_threshold=DEFAULT_DISTANCE_THRESHOLD,
        )

        t_start = time.perf_counter()
        results = run_dataset(cases, retriever, runtime)
        wall_clock_time = time.perf_counter() - t_start

        store.close()

    # ── Per-case report ──────────────────────────────────────────────────────
    print(f"\n{DIVIDER}\nRESULTS\n{DIVIDER}")
    for r in results:
        status = "PASS" if r.passed else "FAIL"
        print(f"\n[{status}] {r.case.id}  ({r.case.category})")
        print(f"  question:           {r.case.question!r}")
        print(f"  expected_type:      {r.case.expected_type}")
        print(f"  actual_type:        {r.actual_type}")
        print(f"  retrieved_chunks:   {r.retrieved_chunk_count}")
        print(f"  retrieved_sources:  {list(r.actual_sources)}")
        print(f"  type_match:         {r.type_match}")
        print(f"  sources_match:      {r.sources_match}")
        if r.keyword_match is not None:
            print(f"  keyword_match:      {r.keyword_match}  (informational, not a pass/fail gate)")
        if r.case.required_keywords:
            print(f"  required_keywords_ok: {r.required_keywords_ok}  (required: {list(r.case.required_keywords)})")
        print(f"  forbidden_phrase_hits: {list(r.forbidden_phrase_hits)}")
        print(f"  has_visible_think_tags: {r.has_visible_think_tags}")
        print(f"  answer_word_count:  {r.answer_word_count}  (length_ok={r.length_ok})")
        print(f"  repeated_blocks:    {len(r.repeated_blocks)} found")
        print(f"  retrieval_time:     {r.retrieval_time:.3f}s")
        print(
            "  generation_time:    "
            + (f"{r.generation_time:.3f}s" if r.generation_time is not None else "N/A (LLM not called)")
        )
        print(f"  total_time:         {r.total_time:.3f}s")
        print(f"  answer: {r.answer}")

    # ── Guard check: unrelated questions never reach the LLM ────────────────
    print(f"\n{DIVIDER}\nGuard check — unrelated questions blocked before any LLM call\n{DIVIDER}")
    for r in results:
        if r.case.category == "unrelated":
            blocked = r.generation_time is None
            print(f"  {r.case.id}: blocked_by_retrieval={blocked}")

    # ── Summary ───────────────────────────────────────────────────────────────
    summary = summarize(results)
    print(f"\n{DIVIDER}\nSUMMARY\n{DIVIDER}")
    print(f"  total:              {summary['total']}")
    print(f"  passed:             {summary['passed']}")
    print(f"  failed:             {summary['failed']}")
    print(f"  type_mismatches:            {summary['type_mismatches']}")
    print(f"  source_mismatches:          {summary['source_mismatches']}")
    print(f"  keyword_mismatches:         {summary['keyword_mismatches']} (informational only)")
    print(f"  required_keyword_failures:  {summary['required_keyword_failures']}")
    print(f"  forbidden_phrase_failures:  {summary['forbidden_phrase_failures']}")
    print(f"  repetition_failures:        {summary['repetition_failures']}")
    print(f"  visible_think_tag_failures: {summary['visible_think_tag_failures']}")
    print(f"  length_failures:            {summary['length_failures']}")
    print(f"  wall_clock_time:            {wall_clock_time:.3f}s")

    return 1 if summary["failed"] > 0 else 0


if __name__ == "__main__":
    sys.exit(main())

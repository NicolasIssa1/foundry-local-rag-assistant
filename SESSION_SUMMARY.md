# Session Summary — 2026-07-25 Checkpoint (M6 complete, submission audit)

## Project status

- **M1–M5** — complete (ingestion, chunking, embeddings, vector store,
  retrieval, prompt builder, Foundry Local SDK integration).
- **M6** (CLI/quality polish) — source display, relevance filtering,
  chat-model benchmarking/selection, think-block suppression, index-only
  loading, CLI usability/error handling, and the evaluation dataset +
  script are all done. A native indexing crash and a chunk-store/FAISS
  ID desync bug were found and fixed in a later repair session (see
  below).
- **Known open issue:** the strengthened evaluation's personally
  verified result is **9/11** — two answers exceed the 150-word length
  limit. The project is **not** considered finally submission-ready
  until this is resolved; see "Manual QA checkpoint" below and
  "Remaining work" at the end of this file.

## What we completed this session

1. **Chat-model benchmark — committed as `6e9a393`.**
   Compared baseline `qwen2.5-0.5b` against candidate `qwen3-1.7b` using
   identical retrieval results, prompts, embedding model
   (`qwen3-embedding-0.6b`), and relevance threshold (1.25) for both.
   `qwen3-1.7b` was more grounded and factually precise; `qwen2.5-0.5b`
   fabricated an unsupported "API Key endpoint" and gave an internally
   contradictory answer on semantic search.
2. **Think-block suppression — committed as `33c00b2`.**
   `src/llm/think_filter.py` (`ThinkBlockFilter` + `filter_think_stream()`)
   strips `<think>...</think>` blocks from a live token stream, correctly
   handling tags split across arbitrary chunk boundaries. Wired into
   `src/pipeline/query.py`.
3. **qwen3-1.7b promoted to default chat model — committed as `4be7208`.**
   `DEFAULT_CHAT_ALIAS` in `src/llm/client.py` changed from
   `qwen2.5-0.5b` to `qwen3-1.7b`. `DEFAULT_EMBED_ALIAS`
   (`qwen3-embedding-0.6b`) unchanged.
4. **Index-only commands no longer load the chat model — committed as
   `8a30c71`.** `FoundryRuntime(load_chat=False)` used by `main.py
   index`; query commands are unaffected.
5. **Documentation checkpoints — committed as `491377c`, `b2237e4`,
   `77ce4af`.** `SESSION_SUMMARY.md` and `README.md` brought up to date
   with the model change, think-block suppression, and lazy loading;
   README rewritten to match the actual implemented structure (it had
   drifted from the original M1 scaffold text).
6. **CLI usability and error handling — committed as `07ffa44`.**
   `main.py` now fails fast (before loading any model) on a missing
   `--data-dir` or missing `--index-dir`, rejects an empty question via
   an argparse-level validator, handles `Ctrl-C` cleanly (exit 130), and
   has descriptive `--help` text with examples for both subcommands. 9
   new tests in `tests/test_main.py`.
7. **Reusable evaluation dataset + script — committed as `b9d66a3`.**
   `src/evaluation/` (`dataset.py`, `scoring.py`, `runner.py`) plus
   `src/evaluation/eval_dataset.json` (11 cases: 5 grounded, 3 paraphrased
   grounded, 3 unrelated/should-be-refused) and `scripts/evaluate_rag.py`,
   which runs the dataset through the real retriever/threshold/prompt
   builder/chat model and reports per-case + summary results. 29 new
   tests in `tests/test_evaluation.py`.
8. **Final submission audit — this checkpoint.** Verified no secrets,
   `.env` files, `.venv/`, model caches, or generated index/log files are
   tracked; removed the "TEMPORARY... not intended to be committed"
   docstring language from `scripts/benchmark_chat_models.py` (it has
   real, cited submission value) and fixed a stale `qwen2.5-0.5b`
   docstring reference in `scripts/demo_m5.py`; brought `README.md`'s
   test count and Evaluation section up to date and consolidated the
   quick-start workflow (clone → venv → install → index → query → test →
   evaluate) into one place.
9. **Current test baseline: 327 passed, 1 skipped, 0 failed.**
10. **Latest real evaluation run: 11/11 passed** — 0 type mismatches, 0
    source mismatches, 0 keyword mismatches; all 3 unrelated questions
    were blocked before any LLM call. ~40.8s wall-clock for all 11 cases
    on the development machine (not a portable timing benchmark).

## Manual QA checkpoint — 2026-07-25 (later same day)

A repair session fixed a native segmentation fault and a chunk-store ID
desync bug (see `git log` — commits `0600c8a` and `1c17d0c`), added
generation safeguards, and strengthened the evaluation with objective
length/repetition/forbidden-phrase/think-tag/required-keyword checks.
After that session, the following was personally verified by hand:

- **Full test suite: 371 passed, 1 skipped, 0 failed.**
- **Fresh indexing succeeded twice consecutively** (`python main.py
  index`, run back-to-back) — the previously observed segmentation
  fault is **confirmed fixed**. Both runs produced **7 chunks**.
- A grounded query (`python main.py query "What is
  retrieval-augmented generation?"`) worked correctly and displayed
  both `sample.txt` and `sample.md` as sources.
- A vector-search query correctly described **FAISS IndexFlatL2** and
  **L2 distance** — the previously observed false "cosine similarity"
  claim did not occur in this manual check.
- The unrelated "What is the weather in Beirut today?" question was
  refused, with **0 retrieved chunks** — the chat model was not called.
- **No visible `<think>` tags** occurred in any manual check.
- The repository remained **clean** throughout.
- The strengthened evaluation (`python scripts/evaluate_rag.py`)
  correctly detected **two remaining answer-length failures**:
  - **Current personal evaluation result: 9/11 passed.**
  - Both failures were **length-only** (no repetition, no forbidden
    phrases, no think-tag leaks):
    - `rag-definition`: **179 words** (limit: 150)
    - `rag-definition-paraphrase`: **154 words** (limit: 150)
  - **0 type mismatches, 0 source mismatches, 0 forbidden-phrase
    failures, 0 repetition failures, 0 visible-think-tag failures.**

**Next session priority:** deterministic concise streamed output —
without weakening the 150-word evaluation limit. The mechanical bugs
(crash, ID desync, uncontrolled repetition, false-claim root cause) are
fixed; the model still occasionally exceeds the length target on some
questions, and that's the remaining open item before this project can
be called finally submission-ready.

## Real CLI validation (with the current defaults)

`python main.py query "What is retrieval-augmented generation?"` — grounded,
no `<think>` content, real Sources section:

```
**Retrieval-Augmented Generation (RAG)** is a technique used in AI systems to
enhance the quality and relevance of the responses generated by the model.

RAG works by combining the knowledge of a language model with the knowledge
of an external knowledge base. This allows the model to generate responses
that are more accurate, relevant, and up-to-date than responses generated by
a standard language model.

In summary, **RAG** is a technique that enhances the performance of a
language model by integrating external knowledge bases.

Sources:
  [1] sample.txt (page 1)
  [2] sample.md (page 1)
```

`python main.py query "Who won the 2026 FIFA World Cup?"` — retrieval finds
zero chunks above the relevance threshold, so the chat model is **never
called**, and the deterministic refusal is returned instead:

```
[query] Retrieved 0 chunk(s) for: 'Who won the 2026 FIFA World Cup?'
I could not find relevant information in the indexed documents.
```

`python main.py index` — only the embedding model loads (no "Loading chat
model" line):

```
[foundry] Loading embedding model: qwen3-embedding-0.6b
[foundry] qwen3-embedding-0.6b ready.
[index] Loaded 2 document(s). Produced 7 chunk(s).
Done. 7 chunk(s) indexed and saved to data/index
```

## Current repository state

- Working tree is **clean** and **fully synchronized with `origin/main`**
  as of commit `1c17d0c` (`fix(llm): constrain and validate grounded
  generation`); this documentation checkpoint is being committed on top
  of it.
- Environment: Python **3.12.13**, `foundry-local-sdk` **1.2.3**.
- Default models: embedding `qwen3-embedding-0.6b`, chat `qwen3-1.7b`.
- No secrets, `.env` files, `.venv/`, model caches, or generated
  `data/index/` artifacts are tracked in git.

## Remaining work

1. **Deterministic concise streamed output** — the strengthened
   evaluation's personally verified result is 9/11 (two length-only
   failures at 179 and 154 words against a 150-word limit). Next
   session's priority: fix this without weakening the 150-word
   evaluation limit.
2. Five-minute presentation and demo recording — **outside this
   repository**; not blocked by item 1, but final submission should
   wait until item 1 is resolved.

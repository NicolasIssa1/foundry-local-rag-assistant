# Session Summary — 2026-08-08 Final Checkpoint (submission-ready)

## Project status

**Technically complete and ready for submission.** All milestones (M1–M7)
are done, the full automated test suite and the bundled evaluation
dataset both pass consistently, and a fresh-clone reproducibility check
succeeded. The only work remaining is outside this repository's code —
see [Remaining work](#remaining-work) below.

## Final architecture

- **Embedding model:** `qwen3-embedding-0.6b` (Microsoft Foundry Local)
- **Chat model:** `qwen3-1.7b` (Microsoft Foundry Local)
- **Runtime:** Microsoft Foundry Local SDK (`foundry-local-sdk` 1.2.3),
  managed via `FoundryRuntime` (`src/llm/client.py`)
- **Vector index:** FAISS `IndexFlatL2` — squared L2 distance
- **Relevance threshold:** `DEFAULT_DISTANCE_THRESHOLD = 1.25`
- **Chunk metadata:** SQLite (`data/index/chunks.db`)
- **Document ingestion:** `.txt` and `.md`
- **Interface:** Python CLI (`main.py index` / `main.py query`)
- **Local-only confirmed:** no OpenAI or Anthropic cloud API key is
  required by the application; the `openai` Python package is used only
  as a generic OpenAI-compatible client pointed at Foundry Local's local
  endpoint (`http://localhost:5272`). Embedding generation, chat
  generation, FAISS search, and SQLite metadata all run on-device.

## Completed milestones

| # | Milestone | Status |
|---|---|---|
| M1 | Project scaffold and repository setup | ✅ Complete |
| M2 | Document ingestion pipeline (load, clean, chunk) | ✅ Complete |
| M3 | Embedding pipeline and FAISS vector store | ✅ Complete |
| M4 | Retrieval and prompt construction | ✅ Complete |
| M5 | Foundry Local LLM integration and end-to-end query | ✅ Complete |
| M6 | CLI/quality polish — source display, relevance filtering, chat-model benchmarking, think-block suppression, lazy model loading, CLI usability/error handling, evaluation dataset + script | ✅ Complete |
| M7 | Grounding hardening — deterministic generation settings, answer finalization, duplicate/repetition removal, concise-output enforcement, unsupported/missing retrieval-metric guard | ✅ Complete |

## Major issues found and fixed (across all sessions)

1. **Native segmentation fault during `main.py index`** — a
   ctypes/libffi callback-marshaling issue when the Foundry Local SDK's
   native layer invoked a Python progress callback from a background
   thread. Fixed by not passing a progress callback, taking the SDK's
   synchronous code path instead (commit `0600c8a`).
2. **Chunk-store/FAISS ID desync** — could serve stale chunk text after
   re-indexing (commit `1c17d0c`).
3. **Uncontrolled generation** — extreme repetition and a leaked closing
   `</think>` tag, fixed with `max_tokens`, `presence_penalty`, and a
   strengthened system prompt (commit `1c17d0c`).
4. **False "cosine similarity" claim** — root-caused to generic
   textbook language in the indexed sample corpus itself (a grounded
   model can only state what's retrievable); corrected the corpus and
   added `src/evaluation/safety_checks.py` as an independent,
   deterministic detector (commit `1c17d0c`).
5. **Run-to-run non-determinism** — `temperature=0.0` was tried first
   and rejected (it silently disabled `presence_penalty` on this backend
   and produced a degenerate empty-answer loop on a real question);
   `temperature=0.1` with a fixed `random_seed=0` restores both
   anti-repetition behavior and real determinism (commit `ed5b78d`).
6. **Answer-length overruns** — two evaluation cases (179 and 154 words
   against a 150-word limit) with no repetition, forbidden-phrase, or
   think-tag involvement. Fixed with `src/llm/answer_finalizer.py`
   (deterministic whitespace/dedup/length-limiting pass applied to every
   answer) (commit `ed5b78d`).
7. **Unsupported/missing retrieval-metric mentions surviving
   finalization** — `src/llm/metric_guard.py` added to detect an
   affirmed-but-unsupported metric claim (distinguishing affirmation
   from negation, e.g. "...L2 rather than cosine similarity...") and to
   detect a supported, on-topic metric the model omitted, triggering one
   corrective regeneration plus an unconditional deterministic strip as
   a final backstop (commit `49705ab`).

## Final test count

**470 passed, 1 skipped, 0 failed** — `python -m pytest`, ~1s runtime.
Covers ingestion, chunking, embeddings, the vector store,
retrieval/relevance filtering, prompt building, the query pipeline, the
think-block filter, answer finalization, the metric guard,
`FoundryRuntime`'s lazy chat-model loading, the CLI, and the evaluation
harness.

## Final evaluation: 11/11, three consecutive stable runs

`python scripts/evaluate_rag.py` was run three times consecutively
against the real index immediately before final acceptance, with
identical results each time:

| Run | Passed | Type/source mismatches | Forbidden-phrase / repetition / think-tag / length failures | Wall-clock |
|---|---|---|---|---|
| 1 | 11/11 | 0 / 0 | 0 / 0 / 0 / 0 | 56.06s |
| 2 | 11/11 | 0 / 0 | 0 / 0 / 0 / 0 | 54.39s |
| 3 | 11/11 | 0 / 0 | 0 / 0 / 0 / 0 | 54.74s |

All 3 "unrelated" cases (`unrelated-world-cup`, `unrelated-weather`,
`unrelated-capital`) were confirmed blocked before any LLM call in every
run. One `keyword_mismatch` (informational only, never a pass/fail gate)
recurred consistently on `foundry-local-serving` across all three runs —
expected free-form phrasing variance, not a defect.

## Final clean-clone acceptance result

A fresh `git clone` of the GitHub repository into a temporary directory
(outside the working project), with a brand-new Python 3.12 virtual
environment and `pip install -r requirements.txt` from a clean cache,
reproduced:

- All core modules import successfully (`main`, `src.pipeline.*`,
  `src.llm.client`, `src.vectorstore.*`, etc.)
- **470 passed, 1 skipped, 0 failed** — identical to the working copy.

The temporary clone was removed afterward; the real project directory
was not modified.

## Repository hygiene result

**Passed.** `git ls-files` (66 tracked files) contains no `.venv/`,
`__pycache__/`, `.pytest_cache/`, generated `data/index/` artifacts,
downloaded model caches, `.env` files, API keys, or `.DS_Store`. The
sole tracked file under `models/` is the `.gitkeep` sentinel. Working
tree was clean before and after the full acceptance test.

## Real CLI validation (current defaults)

`python main.py query "What is retrieval-augmented generation?"` —
grounded, no `<think>` content, real Sources section:

```
Retrieval-Augmented Generation (RAG) is an AI architecture that enhances
a language model's responses by supplying it with relevant information
retrieved from an external knowledge base at query time.

Sources:
  [1] sample.txt (page 1)
  [2] sample.md (page 1)
```

`python main.py query "Who won the 2026 FIFA World Cup?"` — retrieval
finds zero chunks above the relevance threshold, so the chat model is
**never called**, and the deterministic refusal is returned instead:

```
[query] Retrieved 0 chunk(s) for: 'Who won the 2026 FIFA World Cup?'
I could not find relevant information in the indexed documents.
```

`python main.py index` — only the embedding model loads (no "Loading
chat model" line):

```
[foundry] Loading embedding model: qwen3-embedding-0.6b
[foundry] qwen3-embedding-0.6b ready.
[index] Loaded 2 document(s). Produced 7 chunk(s).
Done. 7 chunk(s) indexed and saved to data/index
```

## Current repository state

- Working tree **clean**, `main` **synchronized with `origin/main`** at
  commit `49705ab` as of this documentation checkpoint.
- Environment: Python **3.12.13**, `foundry-local-sdk` **1.2.3**.
- Default models: embedding `qwen3-embedding-0.6b`, chat `qwen3-1.7b`.
- No secrets, `.env` files, `.venv/`, model caches, or generated
  `data/index/` artifacts are tracked in git.

## Remaining work

No core development work remains. What's left is outside this
repository's code:

1. **Submission packaging.**
2. **5-minute presentation/demo recording.**
3. **Certificate submission.**

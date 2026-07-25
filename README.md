# Foundry Local RAG Assistant

> A fully offline Retrieval-Augmented Generation (RAG) system powered by
> [Microsoft Foundry Local](https://github.com/microsoft/foundry-local).
> Embeddings, retrieval, and chat generation all run on-device — no cloud
> API keys, no data egress.

---

## Overview

This project answers questions grounded in your own local documents,
end-to-end, without any network dependency after the initial model
download:

| Concern | Solution |
|---|---|
| Language model inference | Microsoft Foundry Local (`qwen3-1.7b`) |
| Semantic embeddings | Microsoft Foundry Local (`qwen3-embedding-0.6b`) |
| Vector search | FAISS (in-process, no server) — `IndexFlatL2` |
| Chunk metadata | SQLite (`data/index/chunks.db`) |
| Document ingestion | Plain text (`.txt`) and Markdown (`.md`) |
| Interface | Python CLI (`main.py index` / `main.py query`) |

Key behaviors, beyond a basic "embed, search, generate" loop:

- **Relevance filtering.** Retrieved chunks whose FAISS distance exceeds a
  threshold (`DEFAULT_DISTANCE_THRESHOLD = 1.25`, squared L2) are
  discarded. If *no* chunk passes, the chat model is never called at all —
  the CLI returns a deterministic "I could not find relevant information
  in the indexed documents." message instead of letting the model
  hallucinate an answer to an out-of-scope question.
- **Deterministic source citations.** The `Sources:` section printed after
  every answer is built directly from the retrieved chunks' real metadata
  (filename + page, deduplicated), never from the model's own output.
- **Reasoning-block suppression.** Some chat models (e.g. Qwen3 in
  reasoning mode) prepend a `<think>...</think>` chain-of-thought block to
  every response. `src/llm/think_filter.py` strips it live from the token
  stream — correctly handling tags split across arbitrary chunk
  boundaries — so only the final answer ever reaches the terminal.
- **Lazy chat-model loading.** `main.py index` only downloads/loads the
  embedding model; the chat model is loaded only for `main.py query`.
- **Stable indexing.** A native segmentation fault that could occur
  during `main.py index` (a ctypes/libffi callback marshaling issue in
  the Foundry Local SDK's native layer) and a chunk-store/FAISS ID
  desync bug that could serve stale chunk text after re-indexing have
  both been fixed and verified — see `SESSION_SUMMARY.md` for details.

---

## Architecture

```
 Documents (.txt / .md)
        │
        ▼
 [Load + clean + chunk]  ──────────────►  [Embed chunks]
        (src/ingestion)                    (src/embeddings, Foundry Local)
                                                    │
                                                    ▼
                                         [FAISS index + SQLite chunk store]
                                                (src/vectorstore)
                                                    │
      Question ──► [Embed question] ──► [Top-k retrieval + relevance filter]
                     (src/embeddings)          (src/retrieval)
                                                    │
                              ┌─────────────────────┴─────────────────────┐
                              │ 0 chunks pass threshold                   │
                              ▼                                          │ ≥1 chunk passes
                   Deterministic "not found"                              ▼
                   message (chat model never called)          [Prompt Builder] (src/prompt)
                                                                          │
                                                                          ▼
                                                      [Foundry Local chat model, streamed]
                                                                (src/llm)
                                                                          │
                                                                          ▼
                                                  [Think-block filter] (src/llm/think_filter.py)
                                                                          │
                                                                          ▼
                                                     Final answer + deterministic Sources section
```

---

## Repository Structure

```
foundry-local-rag-assistant/
│
├── main.py                    # Entry point — thin argparse shell, delegates to src/
├── requirements.txt           # Pinned dependencies
├── SESSION_SUMMARY.md         # Running development checkpoint log
│
├── src/
│   ├── ingestion/             # Document loading, cleaning, chunking     (M2)
│   ├── embeddings/            # Foundry Local embedding client wrapper   (M3, M5)
│   ├── vectorstore/           # FAISS index + SQLite chunk metadata      (M3)
│   ├── retrieval/             # Top-k similarity search + relevance filter (M4, M6)
│   ├── prompt/                # Prompt templates, builder, source formatting (M4, M6)
│   ├── llm/                   # FoundryRuntime SDK lifecycle + think-block filter (M5, M6)
│   ├── pipeline/               # index/query orchestration used by main.py (M5)
│   └── evaluation/            # Eval dataset schema, scoring, runner       (M6)
│
├── scripts/                   # Demo, benchmark, and evaluation scripts (not part of the app)
│   ├── demo_m4.py             # Offline pipeline demo (mock embeddings)
│   ├── demo_m5.py             # Live end-to-end demo with real Foundry Local models
│   ├── benchmark_chat_models.py  # Reusable qwen2.5-0.5b vs qwen3-1.7b benchmark
│   └── evaluate_rag.py        # Runs src/evaluation/eval_dataset.json against the real index
│
├── data/                      # Source documents to index
│   └── index/                 # Generated FAISS index + SQLite store (git-ignored)
│
├── models/                    # Downloaded model artifacts (git-ignored)
├── notebooks/                 # Exploratory Jupyter notebooks
├── docs/                      # Architecture diagrams, design notes
│
└── tests/                     # Automated test suite (pytest)
```

---

## Quickstart (macOS)

### Prerequisites

- **Python 3.12** (this project is developed and tested against 3.12.13;
  `foundry-local-sdk` requires 3.11+)
- [Microsoft Foundry Local](https://github.com/microsoft/foundry-local) installed
  and running (see that repo's own installation instructions)
- `pip`

### Installation

```bash
git clone <your-repo-url>
cd foundry-local-rag-assistant

# Create a Python 3.12 virtual environment
python3.12 -m venv .venv
source .venv/bin/activate

pip install -r requirements.txt
```

> Windows/Linux also work — swap the activation line for
> `.\.venv\Scripts\activate` on Windows. `foundry-local-sdk`'s WinML
> variant is selected automatically via the platform guard in
> `requirements.txt`.

### Index your documents

```bash
# Currently supported formats: .txt and .md
cp ~/my-notes/*.md data/

python main.py index
```

On first run this also downloads and caches the embedding model
(`qwen3-embedding-0.6b`); subsequent runs reuse the cache. Only the
embedding model is loaded for indexing — the chat model is not touched.

### Query

```bash
python main.py query "What is retrieval-augmented generation?"
```

This loads both the embedding model and the chat model
(`qwen3-1.7b` by default), streams the answer to the terminal with any
`<think>` reasoning stripped, and prints a `Sources:` section listing the
documents/pages the answer was grounded in.

### Run the tests

```bash
pytest
```

### Run the evaluation

```bash
# Requires an index built with `python main.py index` first
python scripts/evaluate_rag.py
```

Runs the bundled evaluation dataset (`src/evaluation/eval_dataset.json`)
against the real index using the same retriever, threshold, prompt
builder, and chat model as the CLI — see [Evaluation](#evaluation) below
for what it checks and the latest recorded result.

### CLI reference

```
python main.py index [--data-dir DATA_DIR] [--index-dir INDEX_DIR]
python main.py query "<question>" [--index-dir INDEX_DIR] [--k K]
```

| Flag | Default | Meaning |
|---|---|---|
| `--data-dir` | `data` | Directory of `.txt`/`.md` files to index |
| `--index-dir` | `data/index` | Where the FAISS index + SQLite store live |
| `--k` | `5` | Number of chunks to retrieve per query |

### Example: a grounded answer

```
$ python main.py query "What is retrieval-augmented generation?"
...
Answer:
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

### Example: an out-of-scope refusal

```
$ python main.py query "Who won the 2026 FIFA World Cup?"
...
Answer:
I could not find relevant information in the indexed documents.
```

No `Sources:` section is printed here, and the chat model is never
invoked — retrieval found zero chunks within the relevance threshold, so
the CLI short-circuits straight to the deterministic refusal above.

---

## Model selection

Two Foundry Local chat models were benchmarked head-to-head against the
same retrieval results, prompts, embedding model, and relevance threshold:

| | `qwen2.5-0.5b` (previous default) | `qwen3-1.7b` (current default) |
|---|---|---|
| Grounding | Partial — 1 of 3 answers fabricated an unsupported detail | Strong — all answers traced to retrieved context |
| Factual precision | Mixed (one internally contradictory answer) | High, no contradictions observed |
| Reasoning leakage | None (no reasoning mode) | Emits `<think>...</think>` — suppressed by `src/llm/think_filter.py` |
| Generation time | ~1s / answer | ~5–9s / answer |

`qwen3-1.7b` was promoted to the default (`DEFAULT_CHAT_ALIAS` in
`src/llm/client.py`) because grounded, non-fabricated answers matter more
for this use case than the latency difference. See
`scripts/benchmark_chat_models.py` for the reusable benchmark harness.

---

## Design Decisions

**Why FAISS over a hosted vector database?**
FAISS runs in-process with zero infrastructure overhead. For document
collections under ~1M vectors it delivers millisecond search latency,
which is ideal for a local system.

**Why Microsoft Foundry Local?**
Foundry Local provides a unified OpenAI-compatible REST endpoint for both
chat and embedding models running on the local machine, so the same
client code works against any model Foundry supports.

**Why reject low-relevance results before calling the LLM?**
An LLM given irrelevant context will often answer anyway, inventing
details to fill the gap. Filtering by FAISS distance *before* generation
means off-topic questions ("Who won the 2026 World Cup?") get a
deterministic refusal instead of a hallucinated answer, and the chat
model is never even invoked for them.

**Why strip `<think>` blocks in the streaming layer instead of with a
post-hoc regex?**
The CLI streams tokens live as they're generated; buffering the entire
response first to run a regex would defeat streaming output entirely.
`ThinkBlockFilter` is a small state machine that tracks whether it's
inside a think block and buffers only the handful of characters that
might be part of a split tag, so it works correctly regardless of how the
model's token stream happens to chunk the tags.

**Why keep `main.py` thin?**
The entry point is a dispatch layer only. Business logic lives in `src/`
so every component can be tested and imported independently without
invoking the CLI.

---

## Development Milestones

| # | Milestone | Status |
|---|---|---|
| M1 | Project scaffold and repository setup | ✅ Complete |
| M2 | Document ingestion pipeline (load, clean, chunk) | ✅ Complete |
| M3 | Embedding pipeline and FAISS vector store | ✅ Complete |
| M4 | Retrieval and prompt construction | ✅ Complete |
| M5 | Foundry Local LLM integration and end-to-end query | ✅ Complete |
| M6 | CLI/quality polish — source display, relevance filtering, chat-model benchmarking, think-block suppression, lazy model loading, CLI usability/error handling, evaluation dataset + script | ✅ Complete, with one known open item below |

**Not yet finally submission-ready.** A known evaluation issue remains:
the latest personally verified evaluation run is **9/11**, with 2
answers exceeding the 150-word length limit (see
[Evaluation](#evaluation)). Deterministic concise output is the next
priority, tracked in `SESSION_SUMMARY.md`. The 5-minute
presentation/demo recording (outside this repository) is also still
remaining.

---

## Testing

```bash
pytest
```

As of this checkpoint: **371 passed, 1 skipped, 0 failed** — covering
ingestion, chunking, embeddings, the vector store, retrieval/relevance
filtering, prompt building, the query pipeline, the think-block filter,
`FoundryRuntime`'s lazy chat-model loading, the CLI, and the evaluation
harness.

---

## Evaluation

Beyond unit tests, `scripts/evaluate_rag.py` runs a reusable dataset of
end-to-end cases (`src/evaluation/eval_dataset.json`) through the real
retriever, relevance threshold, prompt builder, and chat model — the same
code path `main.py query` uses.

| Category | Count |
|---|---|
| Grounded (direct questions) | 5 |
| Grounded (paraphrased questions) | 3 |
| Unrelated (should be refused) | 3 |
| **Total** | **11** |

For each case the script checks several **objective, deterministic**
gates — none of them LLM judging:

- Whether the question was correctly grounded vs. refused, and whether
  the expected source document(s) were among the retrieved chunks.
- **Forbidden phrases** — an answer must not contradict this project's
  actual implementation (e.g. claiming "cosine similarity" when
  retrieval actually uses FAISS `IndexFlatL2`/squared L2 distance).
- **Required keywords** — for cases where an exact implementation fact
  matters, that fact must actually appear in the answer.
- **Visible `<think>` tags** — an independent re-check that no raw
  reasoning leaked into the final answer.
- **Answer length** — a per-case or default word-count ceiling (150
  words by default).
- **Repeated blocks** — detects degenerate repetition loops, including
  the pattern where each repeat is prefixed with a different fabricated
  citation number.

An optional `expected_keywords` field is also reported per case, but is
purely informational and never fails a case — free-form LLM phrasing can
vary even for a fully correct answer.

**Latest personally verified local run: 9/11 passed.** The 2 failures
were both **length-only** — `rag-definition` (179 words) and
`rag-definition-paraphrase` (154 words) each exceeded the 150-word
limit; neither involved repetition, a forbidden phrase, or a visible
think-tag leak. Across that run: 0 type mismatches, 0 source mismatches,
0 forbidden-phrase failures, 0 repetition failures, 0 visible-think-tag
failures. **Deterministic concise output is the next open item** — see
[Limitations and future improvements](#limitations-and-future-improvements)
and `SESSION_SUMMARY.md`.

---

## Limitations and future improvements

- **Concise output is not yet consistent (next priority).** The
  strengthened evaluation enforces a 150-word answer limit; the latest
  personally verified run passed 9/11, with 2 answers exceeding that
  limit (179 and 154 words). No repetition, forbidden phrases, or
  think-tag leaks were involved — this is purely a length-consistency
  issue. Fixing this without weakening the 150-word evaluation limit is
  the next session's priority (see `SESSION_SUMMARY.md`).
- **Document formats.** Only `.txt` and `.md` are currently wired into
  the ingestion loader (`src/ingestion/loader.py`), even though
  `pymupdf` and `python-docx` are already project dependencies. PDF and
  DOCX support is not yet implemented.
- **Small evaluation corpus.** The evaluation dataset (11 cases) is
  scoped to the 2 sample documents shipped with this repo; a larger or
  different document set would need its own dataset with re-verified
  `expected_sources`.
- **Single-machine, single-user design.** There is no concurrency
  control around the SQLite chunk store or FAISS index — this is a
  local CLI tool, not a multi-user service.
- **Fixed relevance threshold.** `DEFAULT_DISTANCE_THRESHOLD = 1.25` is
  a single global constant tuned against the sample documents; a larger
  or more varied corpus may need retuning.

---

## Contributing

This project follows standard Python conventions:

- Format: `black` + `isort`
- Lint: `flake8`
- Tests: `pytest`

Run all checks before committing:

```bash
black src/ tests/ main.py
isort src/ tests/ main.py
flake8 src/ tests/ main.py
pytest
```

---

## License

MIT

---

*Built with Microsoft Foundry Local · Microsoft Summer School 2026*

# Foundry Local RAG Assistant

> A production-quality, fully offline Retrieval-Augmented Generation (RAG) system
> powered by [Microsoft Foundry Local](https://github.com/microsoft/foundry-local).
> All inference and embedding runs on-device — no cloud API keys, no data egress.

---

## Overview

Enterprise AI workloads increasingly require **data sovereignty** — the guarantee that
sensitive documents never leave a controlled environment. This project demonstrates how
to build a complete RAG pipeline that satisfies that constraint by running every
component locally:

| Concern | Solution |
|---|---|
| Language model inference | Microsoft Foundry Local |
| Semantic embeddings | Foundry Local embedding endpoint |
| Vector search | FAISS (in-process, no server) |
| Document ingestion | PyMuPDF · python-docx · markdown |
| Interface | Rich terminal CLI |

The result is a system that can answer questions grounded in your own documents with
**zero network dependency** after the initial model download.

---

## Architecture

```
 ┌──────────────────────────────────────────────────────────┐
 │                        RAG Pipeline                      │
 │                                                          │
 │  Documents          Ingestion           Vector Store     │
 │  (PDF/DOCX/MD) ──► [Chunk + Clean] ──► [FAISS Index]   │
 │                          │                    │          │
 │                    [Embed chunks]       [Embed query]    │
 │                          │                    │          │
 │                          └──── Foundry ────── ┘         │
 │                               Local API                  │
 │                                   │                      │
 │                          Top-k chunk retrieval           │
 │                                   │                      │
 │                          [Prompt Builder]                 │
 │                                   │                      │
 │                          [Foundry Local LLM]             │
 │                                   │                      │
 │                              Final Answer                 │
 └──────────────────────────────────────────────────────────┘
```

---

## Repository Structure

```
foundry-local-rag-assistant/
│
├── main.py                 # Entry point — thin shell, delegates to src/
├── requirements.txt        # Pinned dependencies
├── .gitignore
├── README.md
│
├── src/                    # All application logic
│   ├── __init__.py
│   ├── ingestion/          # Document loading and chunking        (M2)
│   ├── embeddings/         # Embedding model interface            (M3)
│   ├── vectorstore/        # FAISS index management               (M3)
│   ├── retrieval/          # Similarity search and ranking        (M4)
│   ├── prompt/             # Prompt templates and builder         (M4)
│   ├── llm/                # Foundry Local client wrapper         (M5)
│   └── cli/                # Terminal interface                   (M6)
│
├── data/                   # Source documents to index
│
├── models/                 # Downloaded model artifacts (git-ignored)
│
├── notebooks/              # Exploratory Jupyter notebooks
│
├── docs/                   # Architecture diagrams, design notes
│
└── tests/                  # Automated test suite
    └── __init__.py
```

---

## Quickstart

### Prerequisites

- Python 3.10+
- [Microsoft Foundry Local](https://github.com/microsoft/foundry-local) installed and running
- `pip`

### Installation

```bash
# Clone
git clone <your-repo-url>
cd foundry-local-rag-assistant

# Create isolated environment
python -m venv .venv
source .venv/bin/activate        # macOS / Linux
# .\.venv\Scripts\activate       # Windows

# Install dependencies
pip install -r requirements.txt
```

### Index your documents

```bash
# Drop PDFs, DOCX, or Markdown files into data/
cp ~/my-docs/*.pdf data/

# Build the vector index
python main.py index
```

### Query

```bash
python main.py query "What are the key findings in the Q4 report?"
```

---

## Development Milestones

| # | Milestone | Status |
|---|---|---|
| M1 | Project scaffold and repository setup | ✅ Complete |
| M2 | Document ingestion pipeline (load, clean, chunk) | Pending |
| M3 | Embedding pipeline and FAISS vector store | Pending |
| M4 | Retrieval and prompt construction | Pending |
| M5 | Foundry Local LLM integration and end-to-end query | Pending |
| M6 | CLI polish, error handling, and demo preparation | Pending |

---

## Design Decisions

**Why FAISS over a hosted vector database?**
FAISS runs in-process with zero infrastructure overhead. For document collections under
~1 M vectors it delivers millisecond search latency, which is ideal for a local system.

**Why Microsoft Foundry Local?**
Foundry Local provides a unified OpenAI-compatible REST endpoint for both chat and
embedding models running on the local machine. This means the same client code works
against any model Foundry supports, with no vendor-specific SDK changes required.

**Why keep `main.py` thin?**
The entry point is a dispatch layer only. Business logic lives in `src/` so every
component can be tested and imported independently without invoking the CLI.

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
pytest tests/
```

---

## License

MIT

---

*Built with Microsoft Foundry Local · Microsoft Summer School 2026*

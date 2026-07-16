from __future__ import annotations

from pathlib import Path

from ..ingestion.models import Chunk
from .templates import NO_CONTEXT_PLACEHOLDER, RAG_TEMPLATE, SYSTEM_PROMPT


def build(chunks: list[Chunk], question: str) -> str:
    """Assemble retrieved chunks and a question into the final LLM prompt.

    Each chunk is numbered [1], [2], ... and annotated with its source
    filename and page number so the model can cite its answers. The prompt
    is structured as system instruction → numbered context → question.

    Raises ValueError for an empty question.
    """
    if not question or not question.strip():
        raise ValueError("question must be a non-empty string")

    if chunks:
        parts: list[str] = []
        for rank, chunk in enumerate(chunks, start=1):
            filename = Path(chunk.source).name
            header = f"[{rank}] (source: {filename}, page {chunk.page})"
            parts.append(f"{header}\n{chunk.text}")
        context = "\n\n".join(parts)
    else:
        context = NO_CONTEXT_PLACEHOLDER

    return RAG_TEMPLATE.format(
        system_prompt=SYSTEM_PROMPT,
        context=context,
        question=question,
    )


def format_sources(chunks: list[Chunk]) -> str:
    """Render a deduplicated 'Sources' section directly from retrieved chunks.

    Every line comes straight from a Chunk's real metadata (never from the
    model's own output). Chunks are deduplicated by (filename, page) so
    multiple chunks from the same page of the same document collapse into
    one entry; order matches retrieval relevance (first occurrence wins).
    """
    if not chunks:
        return "Sources: (none)"

    seen: set[tuple[str, object]] = set()
    lines: list[str] = []
    for chunk in chunks:
        filename = Path(chunk.source).name if chunk.source else "(unknown source)"
        page = getattr(chunk, "page", None)
        key = (filename, page)
        if key in seen:
            continue
        seen.add(key)

        rank = len(lines) + 1
        if page is not None:
            lines.append(f"  [{rank}] {filename} (page {page})")
        else:
            lines.append(f"  [{rank}] {filename}")

    return "Sources:\n" + "\n".join(lines)

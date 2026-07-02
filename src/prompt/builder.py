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

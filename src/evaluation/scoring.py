"""Pure, deterministic scoring for evaluation cases — no I/O, no models.

Kept separate from runner.py so scoring logic can be unit-tested without
touching FAISS, SQLite, or the Foundry Local SDK.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from ..ingestion.models import Chunk
from .dataset import EvalCase


@dataclass(frozen=True)
class EvalResult:
    case: EvalCase
    actual_type: str
    retrieved_chunk_count: int
    actual_sources: tuple[str, ...]
    answer: str
    retrieval_time: float
    generation_time: float | None
    total_time: float
    type_match: bool
    sources_match: bool
    keyword_match: bool | None  # None when the case declared no expected_keywords

    @property
    def passed(self) -> bool:
        """Objective pass/fail: retrieval type + source membership only.

        keyword_match is deliberately excluded — it checks free-form LLM
        phrasing, which can vary between two equally correct, fully
        grounded answers. type_match and sources_match are both derived
        purely from retrieval against the fixed local index, so they are
        deterministic and reproducible regardless of what the chat model
        says.
        """
        return self.type_match and self.sources_match


def _actual_sources(chunks: list[Chunk]) -> tuple[str, ...]:
    """Deduplicated source filenames, in first-seen (relevance) order."""
    seen: list[str] = []
    for c in chunks:
        name = Path(c.source).name if c.source else "(unknown source)"
        if name not in seen:
            seen.append(name)
    return tuple(seen)


def score_case(
    case: EvalCase,
    chunks: list[Chunk],
    answer: str,
    retrieval_time: float,
    generation_time: float | None,
) -> EvalResult:
    """Score one case against its real retrieval/generation outcome.

    sources_match uses subset semantics: every expected source must appear
    among the retrieved chunks, but additional incidental sources (common
    with a small corpus and k > 1) do not cause a failure.
    """
    actual_type = "grounded" if chunks else "refused"
    actual_sources = _actual_sources(chunks)

    type_match = actual_type == case.expected_type
    sources_match = set(case.expected_sources).issubset(set(actual_sources))

    keyword_match: bool | None = None
    if case.expected_keywords:
        lowered = answer.lower()
        keyword_match = all(kw.lower() in lowered for kw in case.expected_keywords)

    total_time = retrieval_time + (generation_time or 0.0)

    return EvalResult(
        case=case,
        actual_type=actual_type,
        retrieved_chunk_count=len(chunks),
        actual_sources=actual_sources,
        answer=answer,
        retrieval_time=retrieval_time,
        generation_time=generation_time,
        total_time=total_time,
        type_match=type_match,
        sources_match=sources_match,
        keyword_match=keyword_match,
    )


def summarize(results: list[EvalResult]) -> dict:
    """Deterministic aggregate counts over a list of EvalResults."""
    return {
        "total": len(results),
        "passed": sum(1 for r in results if r.passed),
        "failed": sum(1 for r in results if not r.passed),
        "type_mismatches": sum(1 for r in results if not r.type_match),
        "source_mismatches": sum(1 for r in results if not r.sources_match),
        "keyword_mismatches": sum(1 for r in results if r.keyword_match is False),
    }

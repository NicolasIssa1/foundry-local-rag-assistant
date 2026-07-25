"""Pure, deterministic scoring for evaluation cases — no I/O, no models.

Kept separate from runner.py so scoring logic can be unit-tested without
touching FAISS, SQLite, or the Foundry Local SDK.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from ..ingestion.models import Chunk
from .dataset import EvalCase
from .safety_checks import (
    contains_visible_think_tags,
    count_words,
    find_forbidden_phrases,
    find_repeated_blocks,
)

# Default ceiling on answer length. A case may override this via its own
# max_answer_words field. 150 words comfortably covers a genuine multi-point
# bulleted answer (observed real answers: ~25-80 words) while still catching
# runaway/rambling generation.
DEFAULT_MAX_ANSWER_WORDS = 150


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
    forbidden_phrase_hits: tuple[str, ...]
    has_visible_think_tags: bool
    answer_word_count: int
    length_ok: bool
    repeated_blocks: tuple[str, ...]
    required_keywords_ok: bool

    @property
    def passed(self) -> bool:
        """Objective, deterministic pass/fail — retrieval correctness is
        necessary but no longer sufficient.

        Required (hard gates, all derived from plain string checks, never
        LLM judging):
          - type_match / sources_match: retrieval found (or correctly
            didn't find) the right source documents.
          - no forbidden_phrase_hits: the answer doesn't contradict this
            project's actual implementation (e.g. claiming "cosine
            similarity" when retrieval actually uses FAISS/squared L2).
          - no visible think tags: no leaked <think>/</think> content.
          - length_ok: the answer didn't ramble past the length ceiling.
          - no repeated_blocks: no degenerate repetition loop.
          - required_keywords_ok: exact implementation facts that matter
            (e.g. "FAISS") were actually stated, when the case requires it.

        keyword_match (the older, informational expected_keywords field)
        is deliberately EXCLUDED — see its docstring below.
        """
        return (
            self.type_match
            and self.sources_match
            and not self.forbidden_phrase_hits
            and not self.has_visible_think_tags
            and self.length_ok
            and not self.repeated_blocks
            and self.required_keywords_ok
        )


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

    forbidden_phrase_hits = find_forbidden_phrases(answer, case.forbidden_phrases)
    has_visible_think_tags = contains_visible_think_tags(answer)
    answer_word_count = count_words(answer)
    max_words = case.max_answer_words or DEFAULT_MAX_ANSWER_WORDS
    length_ok = answer_word_count <= max_words
    repeated_blocks = find_repeated_blocks(answer)

    required_keywords_ok = True
    if case.required_keywords:
        lowered = answer.lower()
        required_keywords_ok = all(kw.lower() in lowered for kw in case.required_keywords)

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
        forbidden_phrase_hits=forbidden_phrase_hits,
        has_visible_think_tags=has_visible_think_tags,
        answer_word_count=answer_word_count,
        length_ok=length_ok,
        repeated_blocks=repeated_blocks,
        required_keywords_ok=required_keywords_ok,
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
        "forbidden_phrase_failures": sum(1 for r in results if r.forbidden_phrase_hits),
        "repetition_failures": sum(1 for r in results if r.repeated_blocks),
        "visible_think_tag_failures": sum(1 for r in results if r.has_visible_think_tags),
        "length_failures": sum(1 for r in results if not r.length_ok),
        "required_keyword_failures": sum(1 for r in results if not r.required_keywords_ok),
    }

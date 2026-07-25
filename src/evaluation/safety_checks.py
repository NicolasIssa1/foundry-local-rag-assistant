"""Deterministic, objective safety checks for generated answers.

No LLM judging — every check here is a plain string/regex computation over
the answer text, reproducible and independent of model phrasing. Used by
scoring.py to gate evaluation pass/fail on concrete properties (forbidden
phrases, leaked think tags, excessive length, repeated content) rather than
trusting that correct retrieval automatically means a correct answer.
"""
from __future__ import annotations

import re


def find_forbidden_phrases(answer: str, forbidden_phrases: tuple[str, ...]) -> tuple[str, ...]:
    """Return which forbidden phrases (case-insensitive substring match)
    actually appear in the answer, in the caller's original casing."""
    lowered = answer.lower()
    return tuple(p for p in forbidden_phrases if p.lower() in lowered)


def contains_visible_think_tags(answer: str) -> bool:
    """True if a literal <think> or </think> tag leaked into the answer.

    Independent, defense-in-depth check: the streaming ThinkBlockFilter
    (src/llm/think_filter.py) should already prevent this; this re-verifies
    the final answer text as a separate, objective evaluation gate.
    """
    return "<think>" in answer or "</think>" in answer


def count_words(answer: str) -> int:
    return len(answer.split())


def _normalize(text: str) -> str:
    return re.sub(r"\s+", " ", text.strip().lower())


def find_repeated_blocks(answer: str, min_words: int = 6) -> tuple[str, ...]:
    """Return any sentence (>= min_words) that appears more than once,
    verbatim (whitespace/case-normalized), in the answer.

    Catches degenerate repetition loops — e.g. the same sentence repeated
    several times with a different fabricated citation number prefixed to
    each repeat (observed during manual testing: "[17] (source: ...)\\nThe
    RAG pipeline is a two-phase process...", then "[18] (source:
    ...)\\n<identical sentence>", then "[19] ..."). Splitting only on
    sentence punctuation would treat each whole block (label + sentence)
    as one unit and miss the duplicate, since the labels differ — so
    paragraphs are first split on blank lines, then further on newlines
    and sentence punctuation, isolating the label (short, filtered out by
    min_words) from the repeated substance (long enough to be compared).
    """
    units: list[str] = []
    for paragraph in re.split(r"\n\s*\n", answer):
        for line in paragraph.split("\n"):
            for sentence in re.split(r"(?<=[.!?])\s+", line):
                if sentence.strip():
                    units.append(sentence)

    counts: dict[str, int] = {}
    for unit in units:
        norm = _normalize(unit)
        if len(norm.split()) >= min_words:
            counts[norm] = counts.get(norm, 0) + 1

    return tuple(u for u, c in counts.items() if c > 1)

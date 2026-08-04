"""General, corpus-agnostic guard against unsupported similarity/distance
metric claims in a generated answer.

Some local chat models restate a term that appears in retrieved context
even when the context explicitly *negates* it — e.g. "...measures squared
Euclidean (L2) distance between embeddings rather than cosine
similarity..." contains the literal substring "cosine similarity", but the
context is affirmatively stating the opposite metric. A naive "is this
substring anywhere in the context" check is therefore not sufficient.

This module checks, for each known metric named *affirmatively* in the
answer, whether the retrieved context affirmatively states that same
metric in at least one sentence (i.e. without a preceding negation marker
in that sentence). It never inspects question text, case IDs, or any
evaluation-specific data — only the generated answer and the retrieved
context text.
"""
from __future__ import annotations

import re

_METRIC_PATTERNS: dict[str, re.Pattern[str]] = {
    "cosine similarity": re.compile(r"\bcosine\s+similarity\b", re.IGNORECASE),
    "dot product": re.compile(r"\bdot[\s-]product\b", re.IGNORECASE),
    "euclidean distance": re.compile(
        r"\beuclidean\b[^.?!]{0,20}?\bdistance\b", re.IGNORECASE
    ),
    "l2 distance": re.compile(r"\bl2\b[^.?!]{0,10}?\bdistance\b", re.IGNORECASE),
}

_NEGATION_MARKERS = (
    "rather than",
    "instead of",
    "as opposed to",
    "not ",
    "isn't",
    "is not",
    "unlike",
    "never",
    "without using",
)

_CLAUSE_SPLIT = re.compile(r"(?<=[.!?;])\s+")


def _sentences(text: str) -> list[str]:
    """Split into clauses, not just full sentences: a semicolon-joined
    clause (e.g. "...does not use X; it uses Y.") must not let a negation
    in one clause suppress an affirmation in the next."""
    return [s for s in _CLAUSE_SPLIT.split(text) if s.strip()]


def _is_affirmed_in_sentence(metric: str, sentence: str) -> bool:
    match = _METRIC_PATTERNS[metric].search(sentence)
    if not match:
        return False
    preceding = sentence[: match.start()].lower()
    return not any(marker in preceding for marker in _NEGATION_MARKERS)


def find_named_metrics(text: str) -> tuple[str, ...]:
    """Which known metric names are affirmatively mentioned anywhere in
    `text` (i.e. named without an immediately preceding negation marker in
    the same sentence)."""
    hits: list[str] = []
    for metric in _METRIC_PATTERNS:
        for sentence in _sentences(text):
            if _is_affirmed_in_sentence(metric, sentence):
                hits.append(metric)
                break
    return tuple(hits)


def is_metric_supported(metric: str, context: str) -> bool:
    """True if `context` affirmatively states `metric` in at least one
    sentence (not merely mentions it while negating it)."""
    if metric not in _METRIC_PATTERNS:
        return False
    return any(_is_affirmed_in_sentence(metric, sentence) for sentence in _sentences(context))


def find_unsupported_metrics(answer: str, context: str) -> tuple[str, ...]:
    """Metrics named affirmatively in `answer` that `context` does not
    affirmatively support. Empty tuple means every metric claim in the
    answer is grounded in the retrieved context."""
    return tuple(m for m in find_named_metrics(answer) if not is_metric_supported(m, context))

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


def find_metric_mentions(text: str) -> tuple[str, ...]:
    """Which known metric names are mentioned anywhere in `text` at all —
    affirmed, negated, or comparative (e.g. "L2 rather than cosine
    similarity" mentions both "l2 distance" and "cosine similarity"). Unlike
    find_named_metrics(), this does NOT filter out negated mentions —
    mentioning an unsupported metric merely to contrast or rule it out is
    still a mention, and SYSTEM_PROMPT asks the model to omit an
    unsupported metric's name entirely, not just avoid affirming it."""
    return tuple(m for m in _METRIC_PATTERNS if _METRIC_PATTERNS[m].search(text))


def find_unsupported_metrics(answer: str, context: str) -> tuple[str, ...]:
    """Metrics mentioned anywhere in `answer` — affirmed, negated, or
    comparative — that `context` does not affirmatively support.

    An answer must not mention an unsupported metric's name even to
    contrast or explicitly rule it out (e.g. "...measures L2 distance
    rather than cosine similarity..." still names "cosine similarity",
    which the context here only negates — SYSTEM_PROMPT asks for that name
    to be omitted entirely, not merely avoided as an affirmative claim).

    Empty tuple means every metric mentioned in the answer, in any form,
    is affirmatively grounded in the retrieved context.
    """
    return tuple(m for m in find_metric_mentions(answer) if not is_metric_supported(m, context))


# ── Missing-detail guard (conservative, topic-gated) ─────────────────────────
#
# find_unsupported_metrics() above catches an answer *wrongly claiming* a
# metric. It has no counterpart for the opposite failure: an answer that
# omits a metric the context actually, affirmatively supports — e.g. a
# generic "semantic search compares meaning" answer that never mentions the
# specific distance metric the retrieved context states. The functions
# below add that counterpart, deliberately narrow so it never fires on
# unrelated questions (RAG definitions, chunking, serving details, etc.)
# just because some retrieved chunk happens to also contain a metric
# statement — see is_similarity_topic().

_SIMILARITY_TOPIC_PATTERN = re.compile(
    r"\b("
    r"similar(?:ity)?"
    r"|distance"
    r"|nearest[\s-]neighbou?r"
    r"|vector\s+search"
    r"|semantic\s+search"
    r"|match(?:es|ing)?"
    r"|compar(?:e|es|ing|ison)"
    r")\b",
    re.IGNORECASE,
)


def is_similarity_topic(text: str) -> bool:
    """True if `text` (a question or an answer) discusses vector/semantic
    search or similarity/distance behaviour in general terms.

    A conservative, corpus-agnostic topical gate — not tied to any specific
    question, evaluation case, or document. Generic questions about what
    RAG is, what chunking is, or how a service exposes its API do not match
    this pattern, so the missing-metric check below never applies to them.
    """
    return bool(_SIMILARITY_TOPIC_PATTERN.search(text))


def find_missing_supported_metrics(answer: str, context: str) -> tuple[str, ...]:
    """Metrics affirmatively supported by `context` that are not mentioned
    anywhere in `answer` at all.

    Callers should only act on a non-empty result when the question and/or
    answer are actually on-topic for similarity/distance behaviour (see
    is_similarity_topic) — this function alone only has access to the
    answer and context, not the question, so it cannot make that judgement
    by itself.
    """
    mentioned = find_metric_mentions(answer)
    return tuple(
        m for m in _METRIC_PATTERNS
        if m not in mentioned and is_metric_supported(m, context)
    )


# ── Deterministic last-resort safety net (no model call) ────────────────────
#
# generate_answer() allows at most one corrective regeneration, so if the
# model's own retry still mentions an unsupported metric — typically by
# quoting a context sentence that both affirms one metric and negates
# another in the same breath, which happens to satisfy "use the exact
# wording" while still reintroducing the unsupported name — there is no
# budget left to ask the model again. strip_unsupported_metric_mentions()
# is a deterministic, corpus-agnostic text transformation (not a
# regeneration) applied as the final step, guaranteeing the returned answer
# never names an unsupported metric regardless of what the model produced.

_MARKERS_BY_LENGTH_DESC = tuple(sorted((m.strip() for m in _NEGATION_MARKERS), key=len, reverse=True))
_TRAILING_CONNECTORS = " ,;—-"


def strip_unsupported_metric_mentions(answer: str, context: str) -> str:
    """Remove every mention of a metric not affirmatively supported by
    `context`, together with an immediately adjacent negation/comparison
    marker if present (e.g. "...distance rather than cosine similarity"
    becomes "...distance"). Never touches a mention of a metric that IS
    affirmatively supported. Returns `answer` unchanged if nothing is
    unsupported.
    """
    unsupported = find_unsupported_metrics(answer, context)
    if not unsupported:
        return answer

    result = answer
    for metric in unsupported:
        pattern = _METRIC_PATTERNS[metric]
        while True:
            match = pattern.search(result)
            if not match:
                break

            prefix = result[: match.start()].rstrip()
            for marker in _MARKERS_BY_LENGTH_DESC:
                if prefix.lower().endswith(marker):
                    prefix = prefix[: len(prefix) - len(marker)].rstrip()
                    break
            # Drop a stray trailing connector left on the prefix (e.g. "X,
            # rather than Y." -> "X," once "rather than Y" is gone), but
            # deliberately do NOT strip the suffix's leading connector: when
            # the removed span was mid-sentence, that connector (often an
            # em dash or comma) is usually the only thing left separating
            # the kept prefix from the kept suffix, and dropping it too
            # would run the two clauses together.
            prefix = prefix.rstrip(_TRAILING_CONNECTORS).rstrip()

            suffix = result[match.end():].lstrip()

            result = f"{prefix} {suffix}".strip()

    result = re.sub(r"\s+", " ", result)
    result = re.sub(r"\s+([.,!?;])", r"\1", result)
    return result.strip()

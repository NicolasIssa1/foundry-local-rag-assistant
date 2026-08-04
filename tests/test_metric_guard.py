"""Tests for the corpus-agnostic similarity/distance metric guard —
src/llm/metric_guard.py.

Test fixtures below are hand-written, generic sentences about a vector
search system — deliberately NOT copied from src/evaluation/eval_dataset.json
or data/sample.txt, so these tests verify general behaviour rather than
one specific corpus.
"""
from src.llm.metric_guard import (
    find_named_metrics,
    find_unsupported_metrics,
    is_metric_supported,
)

CONTEXT_AFFIRMS_L2 = (
    "This system's vector index uses an L2 index, which measures squared "
    "Euclidean distance between embeddings rather than cosine similarity. "
    "A lower distance means a closer match."
)

CONTEXT_AFFIRMS_L2_DISTANCE = (
    "This system's vector index measures L2 distance between embeddings, "
    "not cosine similarity."
)

CONTEXT_NO_METRIC_MENTIONED = (
    "Documents are split into chunks and converted into embeddings before "
    "being stored in a vector index for later retrieval."
)

CONTEXT_AFFIRMS_COSINE = (
    "This system's vector index uses cosine similarity to rank matches, "
    "not Euclidean distance."
)


# ── find_named_metrics ────────────────────────────────────────────────────────

def test_finds_cosine_similarity_named_in_answer():
    answer = "The system ranks results using cosine similarity."
    assert "cosine similarity" in find_named_metrics(answer)


def test_finds_l2_distance_named_in_answer():
    answer = "Matches are ranked by L2 distance between embeddings."
    assert "l2 distance" in find_named_metrics(answer)


def test_finds_euclidean_distance_named_in_answer():
    answer = "The index computes Euclidean distance between vectors."
    assert "euclidean distance" in find_named_metrics(answer)


def test_finds_dot_product_named_in_answer():
    answer = "Similarity is scored via dot product of the two vectors."
    assert "dot product" in find_named_metrics(answer)


def test_no_metric_named_when_answer_has_none():
    answer = "Documents are split into chunks and embedded before indexing."
    assert find_named_metrics(answer) == ()


def test_negated_metric_mention_is_not_counted_as_named():
    """An answer correctly saying a metric is NOT used must not be treated
    as affirmatively naming that metric."""
    answer = "The system does not use cosine similarity; it uses L2 distance."
    hits = find_named_metrics(answer)
    assert "cosine similarity" not in hits
    assert "l2 distance" in hits


# ── is_metric_supported ───────────────────────────────────────────────────────

def test_l2_distance_supported_when_context_affirms_it():
    assert is_metric_supported("l2 distance", CONTEXT_AFFIRMS_L2_DISTANCE) is True


def test_l2_distance_not_supported_when_context_negates_it():
    context = "This system uses cosine similarity to rank matches, not L2 distance."
    assert is_metric_supported("l2 distance", context) is False


def test_euclidean_distance_supported_when_context_affirms_it():
    assert is_metric_supported("euclidean distance", CONTEXT_AFFIRMS_L2) is True


def test_cosine_similarity_not_supported_when_context_only_negates_it():
    """The literal substring 'cosine similarity' IS present in
    CONTEXT_AFFIRMS_L2, but only inside a negation — must not count as
    support."""
    assert is_metric_supported("cosine similarity", CONTEXT_AFFIRMS_L2) is False


def test_cosine_similarity_supported_when_context_affirms_it():
    assert is_metric_supported("cosine similarity", CONTEXT_AFFIRMS_COSINE) is True


def test_euclidean_distance_not_supported_when_context_negates_it():
    assert is_metric_supported("euclidean distance", CONTEXT_AFFIRMS_COSINE) is False


def test_no_metric_supported_when_context_never_mentions_it():
    assert is_metric_supported("cosine similarity", CONTEXT_NO_METRIC_MENTIONED) is False
    assert is_metric_supported("l2 distance", CONTEXT_NO_METRIC_MENTIONED) is False


def test_unknown_metric_name_is_never_supported():
    assert is_metric_supported("manhattan distance", CONTEXT_AFFIRMS_L2) is False


# ── find_unsupported_metrics (the composed check used by generation.py) ──────

def test_unsupported_metric_detected_end_to_end():
    """Regression for the real observed failure: the model claims cosine
    similarity when the context only mentions it to rule it out."""
    answer = "Vector search in this system relies on cosine similarity."
    hits = find_unsupported_metrics(answer, CONTEXT_AFFIRMS_L2)
    assert hits == ("cosine similarity",)


def test_supported_metric_produces_no_hits():
    """A correct, context-affirmed metric claim must not be flagged."""
    answer = "This system measures Euclidean distance between embeddings."
    assert find_unsupported_metrics(answer, CONTEXT_AFFIRMS_L2) == ()


def test_no_hits_when_answer_names_no_metric():
    answer = "Documents are chunked and embedded before indexing."
    assert find_unsupported_metrics(answer, CONTEXT_AFFIRMS_L2) == ()


def test_multiple_unsupported_metrics_all_detected():
    answer = "The system uses cosine similarity and dot product to rank matches."
    hits = find_unsupported_metrics(answer, CONTEXT_NO_METRIC_MENTIONED)
    assert set(hits) == {"cosine similarity", "dot product"}

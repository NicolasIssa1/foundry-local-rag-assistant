"""Tests for the corpus-agnostic similarity/distance metric guard —
src/llm/metric_guard.py.

Test fixtures below are hand-written, generic sentences about a vector
search system — deliberately NOT copied from src/evaluation/eval_dataset.json
or data/sample.txt, so these tests verify general behaviour rather than
one specific corpus.
"""
from src.llm.metric_guard import (
    find_metric_mentions,
    find_missing_supported_metrics,
    find_named_metrics,
    find_unsupported_metrics,
    is_metric_supported,
    is_similarity_topic,
    strip_unsupported_metric_mentions,
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


# ── is_similarity_topic ────────────────────────────────────────────────────────

def test_similarity_topic_true_for_vector_search_question():
    assert is_similarity_topic("How does vector search work?") is True


def test_similarity_topic_true_for_semantic_search_question():
    assert is_similarity_topic("How does semantic search work?") is True


def test_similarity_topic_true_for_similar_text_question():
    assert is_similarity_topic("How does it find similar text?") is True


def test_similarity_topic_true_for_distance_mention():
    assert is_similarity_topic("What distance measure does it use?") is True


def test_similarity_topic_false_for_rag_definition_question():
    assert is_similarity_topic("What is retrieval-augmented generation?") is False


def test_similarity_topic_false_for_chunking_question():
    assert is_similarity_topic("What is chunking in a RAG pipeline?") is False


def test_similarity_topic_false_for_serving_question():
    assert is_similarity_topic("How does Foundry Local serve models locally?") is False


def test_similarity_topic_checks_answer_text_too():
    """The gate is applied to arbitrary text, not just questions — callers
    may also check the generated answer."""
    answer = "The nearest match is found by comparing vectors."
    assert is_similarity_topic(answer) is True


# ── find_missing_supported_metrics ───────────────────────────────────────────────

def test_missing_supported_metric_detected():
    answer = "Vector search compares the meaning of pieces of text."
    hits = find_missing_supported_metrics(answer, CONTEXT_AFFIRMS_L2_DISTANCE)
    assert "l2 distance" in hits


def test_missing_supported_metrics_empty_when_metric_already_present():
    answer = "This system measures Euclidean distance between embeddings."
    assert find_missing_supported_metrics(answer, CONTEXT_AFFIRMS_L2) == ()


def test_missing_supported_metrics_empty_when_context_supports_nothing():
    answer = "Documents are chunked and embedded before indexing."
    assert find_missing_supported_metrics(answer, CONTEXT_NO_METRIC_MENTIONED) == ()


def test_missing_supported_metrics_ignores_metric_context_only_negates():
    """CONTEXT_AFFIRMS_L2 mentions "cosine similarity" only to rule it out,
    so an answer omitting cosine similarity must not be flagged as missing
    it — only the affirmatively supported metric (L2/Euclidean) counts."""
    answer = "This system's index is a simple lookup table."
    hits = find_missing_supported_metrics(answer, CONTEXT_AFFIRMS_L2)
    assert "cosine similarity" not in hits
    assert "euclidean distance" in hits or "l2 distance" in hits


# ── find_metric_mentions — mention regardless of affirmed/negated ────────────────

def test_find_metric_mentions_includes_affirmative_mention():
    answer = "The system ranks results using cosine similarity."
    assert "cosine similarity" in find_metric_mentions(answer)


def test_find_metric_mentions_includes_negated_mention():
    """Unlike find_named_metrics, a negated mention still counts as a
    mention — mentioning an unsupported metric's name to rule it out is
    still mentioning it."""
    answer = "This system does not use cosine similarity."
    assert "cosine similarity" in find_metric_mentions(answer)


def test_find_metric_mentions_includes_comparative_mention():
    answer = "It uses L2 distance rather than cosine similarity."
    mentions = find_metric_mentions(answer)
    assert "l2 distance" in mentions
    assert "cosine similarity" in mentions


def test_find_metric_mentions_empty_when_no_metric_present():
    answer = "Documents are chunked and embedded before indexing."
    assert find_metric_mentions(answer) == ()


# ── find_unsupported_metrics — now catches negated/comparative mentions too ──────
# (regression coverage for the observed real failure: a corrective retry
# produced "...measures squared Euclidean distance...rather than cosine
# similarity...", which is factually correct but still names an
# unsupported metric — SYSTEM_PROMPT says to omit it entirely.)

def test_A_affirmative_unsupported_mention_requires_correction():
    """A: context says 'L2 rather than cosine similarity'; answer
    affirmatively says 'cosine similarity' -> correction required."""
    answer = "This system ranks results using cosine similarity."
    hits = find_unsupported_metrics(answer, CONTEXT_AFFIRMS_L2_DISTANCE)
    assert "cosine similarity" in hits


def test_B_negated_unsupported_mention_still_requires_correction():
    """B: same context; answer says 'not cosine similarity' -> correction
    still required, because cosine is not affirmatively supported by
    context, regardless of the answer negating it."""
    answer = "This system does not use cosine similarity for ranking."
    hits = find_unsupported_metrics(answer, CONTEXT_AFFIRMS_L2_DISTANCE)
    assert "cosine similarity" in hits


def test_C_only_supported_metric_mentioned_is_accepted():
    """C: same context; answer says 'L2 distance' only -> accepted."""
    answer = "This system uses L2 distance to rank matches."
    hits = find_unsupported_metrics(answer, CONTEXT_AFFIRMS_L2_DISTANCE)
    assert hits == ()


def test_D_supported_metric_named_is_accepted():
    """D: context affirmatively states cosine similarity; answer names
    cosine similarity -> accepted."""
    answer = "This system ranks matches using cosine similarity."
    hits = find_unsupported_metrics(answer, CONTEXT_AFFIRMS_COSINE)
    assert hits == ()


def test_comparative_sentence_example_from_requirements_is_rejected():
    """The exact 'not allowed' examples from the design requirements."""
    for answer in [
        "It uses L2 rather than cosine similarity.",
        "It does not use cosine similarity.",
        "Unlike cosine similarity, it uses L2.",
    ]:
        hits = find_unsupported_metrics(answer, CONTEXT_AFFIRMS_L2_DISTANCE)
        assert "cosine similarity" in hits, f"expected a hit for: {answer!r}"


def test_allowed_answer_example_from_requirements_is_accepted():
    answer = "FAISS IndexFlatL2 measures squared L2 distance."
    hits = find_unsupported_metrics(answer, CONTEXT_AFFIRMS_L2_DISTANCE)
    assert hits == ()


# ── strip_unsupported_metric_mentions — deterministic last-resort net ────────────
# (regression coverage for the real observed failure: even after the one
# allowed corrective retry, the model quoted a full context sentence that
# both affirms Euclidean/L2 distance and negates cosine similarity in the
# same breath — e.g. "...measures squared Euclidean distance between
# embeddings rather than cosine similarity — a lower distance means a
# closer match." — leaving "cosine similarity" in the final answer.)

def test_strip_removes_unsupported_metric_and_its_negation_marker():
    answer = "It measures L2 distance rather than cosine similarity."
    result = strip_unsupported_metric_mentions(answer, CONTEXT_AFFIRMS_L2_DISTANCE)
    assert "cosine similarity" not in result.lower()
    assert "l2 distance" in result.lower() or "l2" in result.lower()


def test_strip_reproduces_the_real_observed_sentence_cleanly():
    answer = (
        "This project's vector index uses FAISS IndexFlatL2, which measures "
        "squared Euclidean distance between embeddings rather than cosine "
        "similarity — a lower distance means a closer match, and only chunks "
        "below a configured distance threshold are kept as relevant."
    )
    result = strip_unsupported_metric_mentions(answer, CONTEXT_AFFIRMS_L2)
    assert "cosine similarity" not in result.lower()
    assert "faiss" in result.lower()
    assert "euclidean" in result.lower()
    # The trailing clause (unrelated to the removed metric) must survive.
    assert "lower distance means a closer match" in result


def test_strip_leaves_supported_metric_answer_unchanged():
    answer = "This system measures squared Euclidean distance between embeddings."
    result = strip_unsupported_metric_mentions(answer, CONTEXT_AFFIRMS_L2)
    assert result == answer


def test_strip_leaves_answer_with_no_metrics_unchanged():
    answer = "Documents are chunked and embedded before indexing."
    result = strip_unsupported_metric_mentions(answer, CONTEXT_NO_METRIC_MENTIONED)
    assert result == answer


def test_strip_handles_standalone_negated_sentence():
    answer = "It does not use cosine similarity."
    result = strip_unsupported_metric_mentions(answer, CONTEXT_AFFIRMS_L2_DISTANCE)
    assert "cosine similarity" not in result.lower()


def test_strip_only_removes_unsupported_metric_not_supported_one():
    """If context supports cosine similarity, stripping must never touch a
    correct, supported mention of it."""
    answer = "This system uses cosine similarity, not Euclidean distance."
    result = strip_unsupported_metric_mentions(answer, CONTEXT_AFFIRMS_COSINE)
    assert "cosine similarity" in result.lower()
    assert "euclidean distance" not in result.lower()

"""Tests for the shared generation pipeline — src/llm/generation.py.

Covers: finalisation is applied to every raw answer, at most one
corrective regeneration fires (either for an unsupported metric claim OR
for a supported-but-omitted metric detail — never both), the
missing-detail guard is conservatively topic-gated so it never pollutes
unrelated answers, and streaming/non-streaming produce identical results
for identical raw model output.
"""
from unittest.mock import MagicMock

from src.llm.generation import generate_answer

CONTEXT_SUPPORTS_L2 = (
    "This system's vector index measures squared Euclidean distance "
    "between embeddings rather than cosine similarity."
)

CONTEXT_NO_METRIC = "Documents are chunked and embedded before indexing."

# A context that affirms a metric, mixed in with unrelated RAG-concept
# sentences — mirrors the real corpus, where the metric statement lives in
# the same document as generic RAG/chunking content the model also
# retrieves for other questions.
CONTEXT_MIXED_WITH_METRIC = (
    "Retrieval-Augmented Generation (RAG) is an AI architecture that "
    "enhances a language model's responses using an external knowledge "
    "base. This system's vector index measures squared Euclidean distance "
    "between embeddings rather than cosine similarity. Documents are split "
    "into overlapping chunks before being embedded and indexed."
)

SIMILARITY_QUESTION = "How does vector search work?"
RAG_DEFINITION_QUESTION = "What is retrieval-augmented generation?"
CHUNKING_QUESTION = "What is chunking in a RAG pipeline?"


def _messages(question: str = SIMILARITY_QUESTION) -> list[dict]:
    return [{"role": "user", "content": question}]


# ── Basic generation (no metric issue) ────────────────────────────────────────

def test_streaming_generation_returns_finalized_answer():
    runtime = MagicMock()
    runtime.stream_chat.return_value = iter(["Chunks are embedded and compared."])
    answer = generate_answer(
        runtime, _messages(), CONTEXT_NO_METRIC, SIMILARITY_QUESTION, stream=True
    )
    assert answer == "Chunks are embedded and compared."


def test_non_streaming_generation_returns_finalized_answer():
    runtime = MagicMock()
    runtime.chat.return_value = "Chunks are embedded and compared."
    answer = generate_answer(
        runtime, _messages(), CONTEXT_NO_METRIC, SIMILARITY_QUESTION, stream=False
    )
    assert answer == "Chunks are embedded and compared."


def test_streaming_calls_stream_chat_not_chat():
    runtime = MagicMock()
    runtime.stream_chat.return_value = iter(["An answer."])
    generate_answer(runtime, _messages(), CONTEXT_NO_METRIC, SIMILARITY_QUESTION, stream=True)
    runtime.stream_chat.assert_called_once()
    runtime.chat.assert_not_called()


def test_non_streaming_calls_chat_not_stream_chat():
    runtime = MagicMock()
    runtime.chat.return_value = "An answer."
    generate_answer(runtime, _messages(), CONTEXT_NO_METRIC, SIMILARITY_QUESTION, stream=False)
    runtime.chat.assert_called_once()
    runtime.stream_chat.assert_not_called()


def test_no_regeneration_when_no_metric_named_and_none_supported():
    runtime = MagicMock()
    runtime.stream_chat.return_value = iter(["Chunks are embedded and compared."])
    generate_answer(runtime, _messages(), CONTEXT_NO_METRIC, SIMILARITY_QUESTION, stream=True)
    assert runtime.stream_chat.call_count == 1


def test_no_regeneration_when_metric_is_supported_by_context():
    runtime = MagicMock()
    runtime.stream_chat.return_value = iter(
        ["This system measures Euclidean distance between vectors."]
    )
    answer = generate_answer(
        runtime, _messages(), CONTEXT_SUPPORTS_L2, SIMILARITY_QUESTION, stream=True
    )
    assert runtime.stream_chat.call_count == 1
    assert "Euclidean distance" in answer


# ── A. Unsupported-claim correction (existing behaviour, unchanged) ─────────────

def test_unsupported_metric_triggers_exactly_one_regeneration():
    runtime = MagicMock()
    runtime.stream_chat.side_effect = [
        iter(["This system relies on cosine similarity."]),
        iter(["This system relies on Euclidean distance."]),
    ]
    answer = generate_answer(
        runtime, _messages(), CONTEXT_SUPPORTS_L2, SIMILARITY_QUESTION, stream=True
    )
    assert runtime.stream_chat.call_count == 2
    assert answer == "This system relies on Euclidean distance."


def test_unsupported_claim_correction_applies_even_off_topic_question():
    """Unsupported-metric protection is unconditional — unlike the
    missing-detail guard, it does not depend on is_similarity_topic()."""
    runtime = MagicMock()
    runtime.stream_chat.side_effect = [
        iter(["This system relies on cosine similarity."]),
        iter(["This system relies on Euclidean distance."]),
    ]
    answer = generate_answer(
        runtime, _messages(RAG_DEFINITION_QUESTION), CONTEXT_SUPPORTS_L2,
        RAG_DEFINITION_QUESTION, stream=True,
    )
    assert runtime.stream_chat.call_count == 2
    assert answer == "This system relies on Euclidean distance."


def test_regeneration_does_not_recurse_even_if_still_unsupported():
    """At most ONE corrective regeneration (one extra model call) — even if
    the retry still mentions an unsupported metric, generate_answer must
    not call the model a third time. The deterministic safety net still
    strips the unsupported mention from what's finally returned."""
    runtime = MagicMock()
    runtime.stream_chat.side_effect = [
        iter(["This system relies on cosine similarity."]),
        iter(["This system still relies on cosine similarity."]),
    ]
    answer = generate_answer(
        runtime, _messages(), CONTEXT_SUPPORTS_L2, SIMILARITY_QUESTION, stream=True
    )
    assert runtime.stream_chat.call_count == 2
    assert "cosine similarity" not in answer.lower()


def test_corrective_regeneration_includes_feedback_message():
    runtime = MagicMock()
    runtime.stream_chat.side_effect = [
        iter(["This system relies on cosine similarity."]),
        iter(["This system relies on Euclidean distance."]),
    ]
    generate_answer(runtime, _messages(), CONTEXT_SUPPORTS_L2, SIMILARITY_QUESTION, stream=True)

    second_call_messages = runtime.stream_chat.call_args_list[1].args[0]
    assert len(second_call_messages) == len(_messages()) + 2
    assert second_call_messages[-2]["role"] == "assistant"
    assert second_call_messages[-1]["role"] == "user"
    assert "cosine similarity" in second_call_messages[-1]["content"]


def test_corrective_regeneration_does_not_guess_a_replacement_metric():
    """generate_answer must never rewrite the text itself with a guessed
    metric — it only asks the model to regenerate."""
    runtime = MagicMock()
    runtime.stream_chat.side_effect = [
        iter(["This system relies on cosine similarity."]),
        iter(["This system does not specify a distance metric."]),
    ]
    answer = generate_answer(
        runtime, _messages(), CONTEXT_SUPPORTS_L2, SIMILARITY_QUESTION, stream=True
    )
    assert "dot product" not in answer.lower()
    assert answer == "This system does not specify a distance metric."


# ── Regression: a negated/comparative unsupported mention still triggers
# correction (the real observed failure — a corrective retry produced
# "...measures squared Euclidean distance...rather than cosine
# similarity...", which is factually correct but still names an
# unsupported metric merely to contrast it). ─────────────────────────────────

def test_negated_unsupported_mention_still_triggers_correction():
    """CONTEXT_SUPPORTS_L2 affirms "Euclidean distance" literally (it never
    says "L2"), so the mock answers here say "Euclidean distance" too —
    otherwise "l2 distance" would itself count as a second, distinct
    unsupported metric under this fixture, which isn't what this test is
    about."""
    runtime = MagicMock()
    runtime.stream_chat.side_effect = [
        iter(["This system measures Euclidean distance rather than cosine similarity."]),
        iter(["This system measures Euclidean distance."]),
    ]
    answer = generate_answer(
        runtime, _messages(), CONTEXT_SUPPORTS_L2, SIMILARITY_QUESTION, stream=True
    )
    assert runtime.stream_chat.call_count == 2
    assert "cosine similarity" not in answer.lower()
    assert answer == "This system measures Euclidean distance."


def test_missing_detail_retry_that_reintroduces_unsupported_mention_is_still_cleaned():
    """End-to-end regression for the real observed live failure: the FIRST
    answer is generic (triggers the missing-detail retry, not the
    unsupported-claim retry), and the retry's own answer quotes a sentence
    that affirms Euclidean distance while also negating cosine similarity
    in the same breath — satisfying "use the exact context wording" while
    still reintroducing an unsupported metric name. The deterministic
    safety net must remove it even though no further model call is made."""
    runtime = MagicMock()
    runtime.stream_chat.side_effect = [
        iter(["Vector search compares the meaning of pieces of text."]),
        iter([
            "This project's vector index uses FAISS IndexFlatL2, which measures "
            "squared Euclidean distance between embeddings rather than cosine "
            "similarity — a lower distance means a closer match."
        ]),
    ]
    answer = generate_answer(
        runtime, _messages(SIMILARITY_QUESTION), CONTEXT_SUPPORTS_L2,
        SIMILARITY_QUESTION, stream=True,
    )
    assert runtime.stream_chat.call_count == 2
    assert "cosine similarity" not in answer.lower()
    assert "euclidean distance" in answer.lower()
    assert "faiss" in answer.lower()


def test_comparative_unsupported_mention_feedback_asks_to_omit_entirely():
    runtime = MagicMock()
    runtime.stream_chat.side_effect = [
        iter(["It does not use cosine similarity; it uses Euclidean distance."]),
        iter(["This system measures Euclidean distance."]),
    ]
    generate_answer(runtime, _messages(), CONTEXT_SUPPORTS_L2, SIMILARITY_QUESTION, stream=True)

    feedback = runtime.stream_chat.call_args_list[1].args[0][-1]["content"]
    assert "cosine similarity" in feedback
    assert "omit" in feedback.lower()


def test_negated_mention_correction_does_not_recurse():
    """Still at most ONE corrective regeneration (one extra model call)
    even for the negated-mention case, if the retry still mentions the
    unsupported metric — and the safety net still removes it from the
    final answer without any further model call."""
    runtime = MagicMock()
    runtime.stream_chat.side_effect = [
        iter(["It uses L2 rather than cosine similarity."]),
        iter(["It still doesn't use cosine similarity."]),
    ]
    answer = generate_answer(
        runtime, _messages(), CONTEXT_SUPPORTS_L2, SIMILARITY_QUESTION, stream=True
    )
    assert runtime.stream_chat.call_count == 2
    assert "cosine similarity" not in answer.lower()


# ── B, F. Missing-detail correction (new) ────────────────────────────────────────

def test_missing_supported_metric_triggers_one_regeneration_on_topic():
    """B: a similarity/vector-search answer that omits a metric the
    context affirmatively supports must trigger exactly one corrective
    retry."""
    runtime = MagicMock()
    runtime.stream_chat.side_effect = [
        iter(["Vector search compares the meaning of pieces of text."]),
        iter(["Vector search measures Euclidean distance between embeddings."]),
    ]
    answer = generate_answer(
        runtime, _messages(SIMILARITY_QUESTION), CONTEXT_SUPPORTS_L2,
        SIMILARITY_QUESTION, stream=True,
    )
    assert runtime.stream_chat.call_count == 2
    assert "Euclidean distance" in answer


def test_missing_detail_regeneration_does_not_recurse():
    """F: even if the retry still omits the supported metric, no third
    call is made."""
    runtime = MagicMock()
    runtime.stream_chat.side_effect = [
        iter(["Vector search compares the meaning of pieces of text."]),
        iter(["Vector search still just compares meaning in general terms."]),
    ]
    answer = generate_answer(
        runtime, _messages(SIMILARITY_QUESTION), CONTEXT_SUPPORTS_L2,
        SIMILARITY_QUESTION, stream=True,
    )
    assert runtime.stream_chat.call_count == 2
    assert answer == "Vector search still just compares meaning in general terms."


def test_missing_detail_feedback_is_generic_and_does_not_name_the_metric():
    """Requirement: feedback for the missing-detail case must stay generic
    (context has a relevant detail; preserve it; don't generalize; don't
    invent) rather than naming the specific metric, unlike the
    unsupported-claim feedback which does name the wrong metric."""
    runtime = MagicMock()
    runtime.stream_chat.side_effect = [
        iter(["Vector search compares the meaning of pieces of text."]),
        iter(["Vector search measures Euclidean distance between embeddings."]),
    ]
    generate_answer(
        runtime, _messages(SIMILARITY_QUESTION), CONTEXT_SUPPORTS_L2,
        SIMILARITY_QUESTION, stream=True,
    )
    feedback = runtime.stream_chat.call_args_list[1].args[0][-1]["content"]
    assert "euclidean" not in feedback.lower()
    assert "l2" not in feedback.lower()
    assert "specific implementation detail" in feedback.lower()
    assert "do not invent" in feedback.lower() or "invent" in feedback.lower()


# ── C, D. Missing-detail guard must NOT fire on unrelated questions ──────────────

def test_rag_definition_answer_not_forced_to_mention_metric():
    """C: even though the retrieved context (mixed corpus text) contains an
    L2 statement, a RAG-definition question/answer must not trigger a
    regeneration just because that statement exists somewhere in context."""
    runtime = MagicMock()
    runtime.stream_chat.return_value = iter(
        ["RAG enhances a language model's responses using an external knowledge base."]
    )
    answer = generate_answer(
        runtime, _messages(RAG_DEFINITION_QUESTION), CONTEXT_MIXED_WITH_METRIC,
        RAG_DEFINITION_QUESTION, stream=True,
    )
    assert runtime.stream_chat.call_count == 1
    assert "euclidean" not in answer.lower()
    assert "l2" not in answer.lower()


def test_chunking_answer_not_forced_to_mention_metric():
    """D: a chunking question/answer must not be forced to mention a
    retrieval metric merely because context also contains one."""
    runtime = MagicMock()
    runtime.stream_chat.return_value = iter(
        ["Documents are split into overlapping chunks before being embedded."]
    )
    answer = generate_answer(
        runtime, _messages(CHUNKING_QUESTION), CONTEXT_MIXED_WITH_METRIC,
        CHUNKING_QUESTION, stream=True,
    )
    assert runtime.stream_chat.call_count == 1
    assert "euclidean" not in answer.lower()
    assert "l2" not in answer.lower()


# ── E. Supported metric wording survives unchanged ────────────────────────────────

def test_supported_metric_wording_survives_unchanged():
    runtime = MagicMock()
    runtime.stream_chat.return_value = iter(
        ["This system measures squared Euclidean distance between embeddings."]
    )
    answer = generate_answer(
        runtime, _messages(SIMILARITY_QUESTION), CONTEXT_SUPPORTS_L2,
        SIMILARITY_QUESTION, stream=True,
    )
    assert runtime.stream_chat.call_count == 1
    assert answer == "This system measures squared Euclidean distance between embeddings."


# ── G. Streaming vs non-streaming consistency ─────────────────────────────────────

def test_streaming_and_non_streaming_agree_given_equivalent_raw_output():
    raw_text = "This system relies on cosine similarity."
    corrected_text = "This system relies on Euclidean distance."

    stream_runtime = MagicMock()
    stream_runtime.stream_chat.side_effect = [iter([raw_text]), iter([corrected_text])]
    stream_answer = generate_answer(
        stream_runtime, _messages(), CONTEXT_SUPPORTS_L2, SIMILARITY_QUESTION, stream=True
    )

    non_stream_runtime = MagicMock()
    non_stream_runtime.chat.side_effect = [raw_text, corrected_text]
    non_stream_answer = generate_answer(
        non_stream_runtime, _messages(), CONTEXT_SUPPORTS_L2, SIMILARITY_QUESTION, stream=False
    )

    assert stream_answer == non_stream_answer
    assert stream_runtime.stream_chat.call_count == non_stream_runtime.chat.call_count == 2


def test_streaming_and_non_streaming_agree_for_missing_detail_path():
    """G: the missing-detail retry path must also behave identically
    between streaming and non-streaming."""
    raw_text = "Vector search compares the meaning of pieces of text."
    corrected_text = "Vector search measures Euclidean distance between embeddings."

    stream_runtime = MagicMock()
    stream_runtime.stream_chat.side_effect = [iter([raw_text]), iter([corrected_text])]
    stream_answer = generate_answer(
        stream_runtime, _messages(SIMILARITY_QUESTION), CONTEXT_SUPPORTS_L2,
        SIMILARITY_QUESTION, stream=True,
    )

    non_stream_runtime = MagicMock()
    non_stream_runtime.chat.side_effect = [raw_text, corrected_text]
    non_stream_answer = generate_answer(
        non_stream_runtime, _messages(SIMILARITY_QUESTION), CONTEXT_SUPPORTS_L2,
        SIMILARITY_QUESTION, stream=False,
    )

    assert stream_answer == non_stream_answer
    assert stream_runtime.stream_chat.call_count == non_stream_runtime.chat.call_count == 2


def test_streaming_applies_same_finalization_as_non_streaming():
    duplicated = "Repeated fact stated twice for testing purposes. Repeated fact stated twice for testing purposes."

    stream_runtime = MagicMock()
    stream_runtime.stream_chat.return_value = iter([duplicated])
    stream_answer = generate_answer(
        stream_runtime, _messages(), CONTEXT_NO_METRIC, SIMILARITY_QUESTION, stream=True
    )

    non_stream_runtime = MagicMock()
    non_stream_runtime.chat.return_value = duplicated
    non_stream_answer = generate_answer(
        non_stream_runtime, _messages(), CONTEXT_NO_METRIC, SIMILARITY_QUESTION, stream=False
    )

    assert stream_answer == non_stream_answer
    assert stream_answer.count("Repeated fact stated twice") == 1


# ── Think-tag filtering integrated into generation ────────────────────────────

def test_think_block_stripped_before_finalization():
    runtime = MagicMock()
    runtime.stream_chat.return_value = iter(
        ["<think>internal reasoning</think>", "Visible answer."]
    )
    answer = generate_answer(
        runtime, _messages(), CONTEXT_NO_METRIC, SIMILARITY_QUESTION, stream=True
    )
    assert answer == "Visible answer."
    assert "internal reasoning" not in answer


# ── H. No evaluation IDs/questions referenced by runtime code ────────────────────

def test_generation_module_does_not_reference_evaluation_dataset():
    """H: generation.py and metric_guard.py must stay corpus-agnostic — no
    evaluation case IDs, no exact evaluation questions, no dataset-specific
    hardcoding of any kind."""
    from pathlib import Path
    import json

    repo_root = Path(__file__).parent.parent
    dataset = json.loads((repo_root / "src/evaluation/eval_dataset.json").read_text())
    forbidden_strings = [case["id"] for case in dataset] + [case["question"] for case in dataset]

    for module_path in ["src/llm/generation.py", "src/llm/metric_guard.py"]:
        source = (repo_root / module_path).read_text()
        for forbidden in forbidden_strings:
            assert forbidden not in source, f"{module_path} hardcodes {forbidden!r}"

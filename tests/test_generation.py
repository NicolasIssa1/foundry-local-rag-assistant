"""Tests for the shared generation pipeline — src/llm/generation.py.

Covers: finalisation is applied to every raw answer, at most one
corrective regeneration fires when an unsupported metric is named, and
streaming/non-streaming produce identical results for identical raw model
output.
"""
from unittest.mock import MagicMock

from src.llm.generation import generate_answer

CONTEXT_SUPPORTS_L2 = (
    "This system's vector index measures squared Euclidean distance "
    "between embeddings rather than cosine similarity."
)

CONTEXT_NO_METRIC = "Documents are chunked and embedded before indexing."


def _messages() -> list[dict]:
    return [{"role": "user", "content": "How does vector search work?"}]


# ── Basic generation (no metric issue) ────────────────────────────────────────

def test_streaming_generation_returns_finalized_answer():
    runtime = MagicMock()
    runtime.stream_chat.return_value = iter(["Chunks are embedded and compared."])
    answer = generate_answer(runtime, _messages(), CONTEXT_NO_METRIC, stream=True)
    assert answer == "Chunks are embedded and compared."


def test_non_streaming_generation_returns_finalized_answer():
    runtime = MagicMock()
    runtime.chat.return_value = "Chunks are embedded and compared."
    answer = generate_answer(runtime, _messages(), CONTEXT_NO_METRIC, stream=False)
    assert answer == "Chunks are embedded and compared."


def test_streaming_calls_stream_chat_not_chat():
    runtime = MagicMock()
    runtime.stream_chat.return_value = iter(["An answer."])
    generate_answer(runtime, _messages(), CONTEXT_NO_METRIC, stream=True)
    runtime.stream_chat.assert_called_once()
    runtime.chat.assert_not_called()


def test_non_streaming_calls_chat_not_stream_chat():
    runtime = MagicMock()
    runtime.chat.return_value = "An answer."
    generate_answer(runtime, _messages(), CONTEXT_NO_METRIC, stream=False)
    runtime.chat.assert_called_once()
    runtime.stream_chat.assert_not_called()


def test_no_regeneration_when_no_metric_named():
    runtime = MagicMock()
    runtime.stream_chat.return_value = iter(["Chunks are embedded and compared."])
    generate_answer(runtime, _messages(), CONTEXT_NO_METRIC, stream=True)
    assert runtime.stream_chat.call_count == 1


def test_no_regeneration_when_metric_is_supported_by_context():
    runtime = MagicMock()
    runtime.stream_chat.return_value = iter(
        ["This system measures Euclidean distance between vectors."]
    )
    answer = generate_answer(runtime, _messages(), CONTEXT_SUPPORTS_L2, stream=True)
    assert runtime.stream_chat.call_count == 1
    assert "Euclidean distance" in answer


# ── Corrective regeneration ───────────────────────────────────────────────────

def test_unsupported_metric_triggers_exactly_one_regeneration():
    runtime = MagicMock()
    runtime.stream_chat.side_effect = [
        iter(["This system relies on cosine similarity."]),
        iter(["This system relies on Euclidean distance."]),
    ]
    answer = generate_answer(runtime, _messages(), CONTEXT_SUPPORTS_L2, stream=True)
    assert runtime.stream_chat.call_count == 2
    assert answer == "This system relies on Euclidean distance."


def test_regeneration_does_not_recurse_even_if_still_unsupported():
    """At most ONE corrective regeneration — even if the retry still
    names an unsupported metric, generate_answer must not call the model
    a third time."""
    runtime = MagicMock()
    runtime.stream_chat.side_effect = [
        iter(["This system relies on cosine similarity."]),
        iter(["This system still relies on cosine similarity."]),
    ]
    answer = generate_answer(runtime, _messages(), CONTEXT_SUPPORTS_L2, stream=True)
    assert runtime.stream_chat.call_count == 2
    assert answer == "This system still relies on cosine similarity."


def test_corrective_regeneration_includes_feedback_message():
    runtime = MagicMock()
    runtime.stream_chat.side_effect = [
        iter(["This system relies on cosine similarity."]),
        iter(["This system relies on Euclidean distance."]),
    ]
    generate_answer(runtime, _messages(), CONTEXT_SUPPORTS_L2, stream=True)

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
    answer = generate_answer(runtime, _messages(), CONTEXT_SUPPORTS_L2, stream=True)
    assert "dot product" not in answer.lower()
    assert answer == "This system does not specify a distance metric."


# ── Streaming vs non-streaming consistency ────────────────────────────────────

def test_streaming_and_non_streaming_agree_given_equivalent_raw_output():
    raw_text = "This system relies on cosine similarity."
    corrected_text = "This system relies on Euclidean distance."

    stream_runtime = MagicMock()
    stream_runtime.stream_chat.side_effect = [iter([raw_text]), iter([corrected_text])]
    stream_answer = generate_answer(stream_runtime, _messages(), CONTEXT_SUPPORTS_L2, stream=True)

    non_stream_runtime = MagicMock()
    non_stream_runtime.chat.side_effect = [raw_text, corrected_text]
    non_stream_answer = generate_answer(
        non_stream_runtime, _messages(), CONTEXT_SUPPORTS_L2, stream=False
    )

    assert stream_answer == non_stream_answer
    assert stream_runtime.stream_chat.call_count == non_stream_runtime.chat.call_count == 2


def test_streaming_applies_same_finalization_as_non_streaming():
    duplicated = "Repeated fact stated twice for testing purposes. Repeated fact stated twice for testing purposes."

    stream_runtime = MagicMock()
    stream_runtime.stream_chat.return_value = iter([duplicated])
    stream_answer = generate_answer(stream_runtime, _messages(), CONTEXT_NO_METRIC, stream=True)

    non_stream_runtime = MagicMock()
    non_stream_runtime.chat.return_value = duplicated
    non_stream_answer = generate_answer(
        non_stream_runtime, _messages(), CONTEXT_NO_METRIC, stream=False
    )

    assert stream_answer == non_stream_answer
    assert stream_answer.count("Repeated fact stated twice") == 1


# ── Think-tag filtering integrated into generation ────────────────────────────

def test_think_block_stripped_before_finalization():
    runtime = MagicMock()
    runtime.stream_chat.return_value = iter(
        ["<think>internal reasoning</think>", "Visible answer."]
    )
    answer = generate_answer(runtime, _messages(), CONTEXT_NO_METRIC, stream=True)
    assert answer == "Visible answer."
    assert "internal reasoning" not in answer

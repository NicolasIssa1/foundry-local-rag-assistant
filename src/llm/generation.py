"""Shared answer-generation pipeline used by both the interactive query
command (src/pipeline/query.py) and the evaluation runner
(src/evaluation/runner.py), so their behaviour never diverges.

Wraps a single raw model call (streaming or non-streaming) with:
  1. live <think> block suppression (ThinkBlockFilter, via filter_think_stream),
  2. deterministic answer finalisation (whitespace/dedup/length — see
     answer_finalizer.finalize_answer),
  3. at most one corrective regeneration if the finalised answer names a
     similarity/distance metric the retrieved context does not
     affirmatively support (see metric_guard).

Contains no reference to any question, corpus, or evaluation case — the
only inputs are the chat messages already built by the caller and the
retrieved context text used to verify metric claims.
"""
from __future__ import annotations

from .answer_finalizer import finalize_answer
from .metric_guard import find_unsupported_metrics
from .think_filter import filter_think_stream


def _raw_generate(runtime, messages: list[dict], stream: bool) -> str:
    if stream:
        return "".join(filter_think_stream(runtime.stream_chat(messages)))
    return "".join(filter_think_stream([runtime.chat(messages)]))


def _correction_messages(
    messages: list[dict], prior_answer: str, unsupported_metrics: tuple[str, ...]
) -> list[dict]:
    metrics_list = ", ".join(f'"{m}"' for m in unsupported_metrics)
    feedback = (
        f"Your previous answer named {metrics_list}, but the context above does "
        "not affirmatively state that. Rewrite the answer using only facts and "
        "terminology explicitly and affirmatively stated in the context. Do not "
        "name any similarity or distance metric unless the context explicitly "
        "states it, and do not guess a replacement metric."
    )
    return messages + [
        {"role": "assistant", "content": prior_answer},
        {"role": "user", "content": feedback},
    ]


def generate_answer(
    runtime, messages: list[dict], context: str, stream: bool = True
) -> str:
    """Generate a finalised, metric-verified answer for one question.

    `context` is the concatenated text of the retrieved chunks the prompt
    in `messages` was built from — used only to verify metric claims, not
    sent to the model again (it's already inside `messages`).
    """
    raw = _raw_generate(runtime, messages, stream)
    answer = finalize_answer(raw)

    unsupported = find_unsupported_metrics(answer, context)
    if unsupported:
        corrective_messages = _correction_messages(messages, answer, unsupported)
        raw_retry = _raw_generate(runtime, corrective_messages, stream)
        answer = finalize_answer(raw_retry)

    return answer

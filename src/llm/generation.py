"""Shared answer-generation pipeline used by both the interactive query
command (src/pipeline/query.py) and the evaluation runner
(src/evaluation/runner.py), so their behaviour never diverges.

Wraps a single raw model call (streaming or non-streaming) with:
  1. live <think> block suppression (ThinkBlockFilter, via filter_think_stream),
  2. deterministic answer finalisation (whitespace/dedup/length — see
     answer_finalizer.finalize_answer),
  3. at most one corrective regeneration, triggered by either of two
     conservative, corpus-agnostic guards (never both at once — see
     generate_answer()):
       - the finalised answer mentions a similarity/distance metric the
         retrieved context does not affirmatively support — whether
         affirmed, negated, or comparative (e.g. "...L2 rather than cosine
         similarity..." still mentions "cosine similarity"; the metric's
         name must be omitted entirely, not merely un-affirmed), or
       - the question/answer is on-topic for similarity/distance behaviour
         and the finalised answer omits a metric the context *does*
         affirmatively support (see metric_guard.is_similarity_topic).
  4. a final, unconditional, deterministic pass (no model call) that
     strips any unsupported metric mention still present after the above —
     e.g. if the one allowed retry itself quotes a context sentence that
     both affirms one metric and negates another (see
     metric_guard.strip_unsupported_metric_mentions).

Contains no reference to any question, corpus, or evaluation case — the
only inputs are the chat messages already built by the caller, the
question text (used only for topic-gating, never echoed or hardcoded),
and the retrieved context text used to verify metric claims.
"""
from __future__ import annotations

from .answer_finalizer import finalize_answer
from .metric_guard import (
    find_missing_supported_metrics,
    find_unsupported_metrics,
    is_similarity_topic,
    strip_unsupported_metric_mentions,
)
from .think_filter import filter_think_stream


def _raw_generate(runtime, messages: list[dict], stream: bool) -> str:
    if stream:
        return "".join(filter_think_stream(runtime.stream_chat(messages)))
    return "".join(filter_think_stream([runtime.chat(messages)]))


def _unsupported_claim_messages(
    messages: list[dict], prior_answer: str, unsupported_metrics: tuple[str, ...]
) -> list[dict]:
    metrics_list = ", ".join(f'"{m}"' for m in unsupported_metrics)
    feedback = (
        f"Your previous answer mentioned {metrics_list}, but the context above "
        "does not affirmatively state that. Rewrite the answer stating only the "
        "metrics the context affirmatively supports. Omit any unsupported "
        f"metric name entirely — do not mention {metrics_list} at all, even to "
        "contrast, compare against, or explicitly rule it out — and do not "
        "guess a replacement metric."
    )
    return messages + [
        {"role": "assistant", "content": prior_answer},
        {"role": "user", "content": feedback},
    ]


def _missing_detail_messages(messages: list[dict], prior_answer: str) -> list[dict]:
    """Feedback for the opposite failure: the context affirmatively states a
    specific, relevant implementation detail that the previous answer
    omitted in favour of a generic explanation. Deliberately does not name
    the missing metric itself, so this stays generic rather than steering
    the model toward one particular word choice.

    Also explicitly warns against reintroducing an unsupported metric while
    adding the supported one: the source sentence for a supported detail
    often also contains the very comparison/negation clause the model must
    NOT repeat (e.g. "...measures squared Euclidean distance...rather than
    cosine similarity..." affirmatively supports Euclidean distance but
    only negates cosine similarity) — quoting that whole sentence verbatim
    would satisfy "use the exact wording" while still reintroducing an
    unsupported metric name. Since this is the last regeneration allowed
    (see generate_answer()), the retried answer is never re-checked, so
    this instruction has to prevent the problem up front rather than catch
    it after the fact.
    """
    feedback = (
        "The context above contains a specific implementation detail that is "
        "directly relevant to this question, and your previous answer did not "
        "include it. Rewrite the answer to state that exact supported detail, "
        "using only the wording and facts given in the context. State only the "
        "detail(s) the context affirmatively supports, on their own — do not "
        "name or contrast against any other similarity or distance metric "
        "unless the context also affirmatively supports that other metric, "
        "even for comparison. Do not substitute a more common or general "
        "explanation, and do not invent any additional details beyond what "
        "the context states."
    )
    return messages + [
        {"role": "assistant", "content": prior_answer},
        {"role": "user", "content": feedback},
    ]


def generate_answer(
    runtime, messages: list[dict], context: str, question: str, stream: bool = True
) -> str:
    """Generate a finalised, metric-verified answer for one question.

    `context` is the concatenated text of the retrieved chunks the prompt
    in `messages` was built from — used only to verify metric claims, not
    sent to the model again (it's already inside `messages`). `question` is
    used only to decide whether the missing-detail guard applies (see
    metric_guard.is_similarity_topic) — never echoed into any message or
    matched against any specific known question.

    At most one corrective regeneration (one extra model call) total: an
    unsupported claim takes priority over a missing detail if (unusually)
    both would apply to the same answer, and the retried answer's model
    output is never re-checked by asking the model again.

    After that, a deterministic, code-only safety net runs unconditionally
    (no model call, so it does not count as a second regeneration): even
    the retried answer may still mention an unsupported metric — e.g. by
    quoting a context sentence that both affirms one metric and negates
    another — so any such mention is stripped before returning. See
    metric_guard.strip_unsupported_metric_mentions().
    """
    raw = _raw_generate(runtime, messages, stream)
    answer = finalize_answer(raw)

    unsupported = find_unsupported_metrics(answer, context)
    missing: tuple[str, ...] = ()
    if not unsupported and (is_similarity_topic(question) or is_similarity_topic(answer)):
        missing = find_missing_supported_metrics(answer, context)

    if unsupported:
        corrective_messages = _unsupported_claim_messages(messages, answer, unsupported)
        raw_retry = _raw_generate(runtime, corrective_messages, stream)
        answer = finalize_answer(raw_retry)
    elif missing:
        corrective_messages = _missing_detail_messages(messages, answer)
        raw_retry = _raw_generate(runtime, corrective_messages, stream)
        answer = finalize_answer(raw_retry)

    return strip_unsupported_metric_mentions(answer, context)

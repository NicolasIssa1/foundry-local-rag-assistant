"""Runs evaluation cases through the real retrieval + chat pipeline.

Mirrors src.pipeline.query.query()'s logic exactly — same Retriever, same
relevance threshold, same prompt builder, same chat model, same
generate_answer() pipeline (think-block filtering, deterministic
finalisation, metric-claim guard) — but keeps retrieval time and
generation time separate for reporting, and never calls the chat model
when nothing passes the relevance threshold. Does not modify or import
anything from main.py/CLI; does not alter application behaviour.
"""
from __future__ import annotations

import time

from ..llm.client import FoundryRuntime
from ..llm.generation import generate_answer
from ..prompt.builder import build
from ..retrieval.retriever import Retriever
from .dataset import EvalCase
from .scoring import EvalResult, score_case

NO_RELEVANT_RESULTS_MESSAGE = "I could not find relevant information in the indexed documents."


def run_case(case: EvalCase, retriever: Retriever, runtime: FoundryRuntime) -> EvalResult:
    """Run a single case: retrieve, then generate only if something passed
    the relevance threshold. The chat model is never called for a case
    that retrieves zero chunks."""
    t0 = time.perf_counter()
    chunks = retriever.retrieve(case.question)
    retrieval_time = time.perf_counter() - t0

    generation_time = None
    if not chunks:
        answer = NO_RELEVANT_RESULTS_MESSAGE
    else:
        prompt = build(chunks, case.question)
        messages = [{"role": "user", "content": prompt}]
        context_text = "\n\n".join(chunk.text for chunk in chunks)
        t1 = time.perf_counter()
        answer = generate_answer(runtime, messages, context_text, case.question, stream=True)
        generation_time = time.perf_counter() - t1

    return score_case(case, chunks, answer, retrieval_time, generation_time)


def run_dataset(
    cases: list[EvalCase], retriever: Retriever, runtime: FoundryRuntime
) -> list[EvalResult]:
    return [run_case(case, retriever, runtime) for case in cases]

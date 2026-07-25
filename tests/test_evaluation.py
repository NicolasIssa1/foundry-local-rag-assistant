"""
Tests for the M6 evaluation harness — src/evaluation/{dataset,scoring,runner}.py.

Covers: dataset schema validation, expected-source matching, grounded and
refusal scoring, zero-retrieval cases skipping the LLM, deterministic
summary counts, and malformed-case handling. No real Foundry SDK or model
is touched — Retriever/FoundryRuntime are mocked throughout.
"""
import json

import pytest
from unittest.mock import MagicMock

from src.evaluation.dataset import EvalCase, InvalidEvalCaseError, load_dataset
from src.evaluation.runner import run_case, run_dataset
from src.evaluation.scoring import score_case, summarize
from src.ingestion.models import Chunk


def _chunk(text: str, source: str, page: int = 1, chunk_index: int = 0) -> Chunk:
    return Chunk(
        text=text, source=source, file_type="txt",
        page=page, chunk_index=chunk_index, start_char=0, end_char=len(text),
    )


def _case(**overrides) -> EvalCase:
    defaults = dict(
        id="case-1",
        question="What is RAG?",
        category="grounded",
        expected_type="grounded",
        expected_sources=("report.txt",),
        expected_keywords=(),
    )
    defaults.update(overrides)
    return EvalCase(**defaults)


def _write_dataset(tmp_path, cases: list[dict]):
    path = tmp_path / "dataset.json"
    path.write_text(json.dumps(cases), encoding="utf-8")
    return path


# ── The real bundled dataset ─────────────────────────────────────────────────

def test_real_bundled_dataset_loads_and_validates():
    cases = load_dataset()
    assert len(cases) >= 3
    categories = {c.category for c in cases}
    assert "grounded" in categories
    assert "unrelated" in categories
    ids = [c.id for c in cases]
    assert len(ids) == len(set(ids))


def test_real_bundled_dataset_has_at_least_one_paraphrase_and_one_unrelated_case():
    cases = load_dataset()
    assert any(c.category == "grounded_paraphrase" for c in cases)
    assert any(c.category == "unrelated" for c in cases)


# ── Dataset schema — valid cases ──────────────────────────────────────────────

def test_load_dataset_parses_minimal_valid_grounded_case(tmp_path):
    path = _write_dataset(tmp_path, [{
        "id": "a", "question": "Q?", "category": "grounded",
        "expected_type": "grounded", "expected_sources": ["doc.txt"],
    }])
    cases = load_dataset(path)
    assert cases == [EvalCase("a", "Q?", "grounded", "grounded", ("doc.txt",), ())]


def test_load_dataset_parses_minimal_valid_refused_case(tmp_path):
    path = _write_dataset(tmp_path, [{
        "id": "b", "question": "Q?", "category": "unrelated", "expected_type": "refused",
    }])
    cases = load_dataset(path)
    assert cases[0].expected_sources == ()
    assert cases[0].expected_keywords == ()


# ── Dataset schema — malformed cases ──────────────────────────────────────────

def test_load_dataset_rejects_missing_required_field(tmp_path):
    path = _write_dataset(tmp_path, [{"id": "a", "question": "Q?", "category": "grounded"}])
    with pytest.raises(InvalidEvalCaseError):
        load_dataset(path)


def test_load_dataset_rejects_empty_question(tmp_path):
    path = _write_dataset(tmp_path, [{
        "id": "a", "question": "   ", "category": "grounded",
        "expected_type": "grounded", "expected_sources": ["doc.txt"],
    }])
    with pytest.raises(InvalidEvalCaseError):
        load_dataset(path)


def test_load_dataset_rejects_invalid_category(tmp_path):
    path = _write_dataset(tmp_path, [{
        "id": "a", "question": "Q?", "category": "bogus",
        "expected_type": "grounded", "expected_sources": ["doc.txt"],
    }])
    with pytest.raises(InvalidEvalCaseError):
        load_dataset(path)


def test_load_dataset_rejects_invalid_expected_type(tmp_path):
    path = _write_dataset(tmp_path, [{
        "id": "a", "question": "Q?", "category": "grounded",
        "expected_type": "maybe", "expected_sources": ["doc.txt"],
    }])
    with pytest.raises(InvalidEvalCaseError):
        load_dataset(path)


def test_load_dataset_rejects_grounded_case_without_expected_sources(tmp_path):
    path = _write_dataset(tmp_path, [{
        "id": "a", "question": "Q?", "category": "grounded", "expected_type": "grounded",
    }])
    with pytest.raises(InvalidEvalCaseError):
        load_dataset(path)


def test_load_dataset_rejects_refused_case_with_expected_sources(tmp_path):
    path = _write_dataset(tmp_path, [{
        "id": "a", "question": "Q?", "category": "unrelated", "expected_type": "refused",
        "expected_sources": ["doc.txt"],
    }])
    with pytest.raises(InvalidEvalCaseError):
        load_dataset(path)


def test_load_dataset_rejects_duplicate_ids(tmp_path):
    case = {
        "id": "dup", "question": "Q?", "category": "grounded",
        "expected_type": "grounded", "expected_sources": ["doc.txt"],
    }
    path = _write_dataset(tmp_path, [case, dict(case, question="Q2?")])
    with pytest.raises(InvalidEvalCaseError):
        load_dataset(path)


def test_load_dataset_rejects_empty_array(tmp_path):
    path = _write_dataset(tmp_path, [])
    with pytest.raises(InvalidEvalCaseError):
        load_dataset(path)


def test_load_dataset_rejects_non_array_json(tmp_path):
    path = tmp_path / "dataset.json"
    path.write_text(json.dumps({"not": "a list"}), encoding="utf-8")
    with pytest.raises(InvalidEvalCaseError):
        load_dataset(path)


def test_load_dataset_rejects_non_object_case(tmp_path):
    path = tmp_path / "dataset.json"
    path.write_text(json.dumps(["just a string"]), encoding="utf-8")
    with pytest.raises(InvalidEvalCaseError):
        load_dataset(path)


# ── Expected-source matching (subset semantics) ────────────────────────────────

def test_sources_match_when_expected_is_subset_of_actual():
    case = _case(expected_sources=("report.txt",))
    chunks = [_chunk("a", "/data/report.txt"), _chunk("b", "/data/guide.md")]
    result = score_case(case, chunks, "answer", 0.1, 0.2)
    assert result.sources_match is True


def test_sources_match_fails_when_expected_source_missing():
    case = _case(expected_sources=("report.txt",))
    chunks = [_chunk("b", "/data/guide.md")]
    result = score_case(case, chunks, "answer", 0.1, 0.2)
    assert result.sources_match is False


def test_sources_match_true_for_refused_case_with_no_expected_sources():
    case = _case(expected_type="refused", expected_sources=())
    result = score_case(case, [], "refused answer", 0.1, None)
    assert result.sources_match is True


# ── Grounded-result scoring ────────────────────────────────────────────────────

def test_grounded_case_passes_when_type_and_sources_match():
    case = _case(expected_type="grounded", expected_sources=("report.txt",))
    chunks = [_chunk("a", "/data/report.txt")]
    result = score_case(case, chunks, "a real answer", 0.05, 1.2)
    assert result.actual_type == "grounded"
    assert result.type_match is True
    assert result.passed is True
    assert result.retrieved_chunk_count == 1
    assert result.total_time == pytest.approx(1.25)


def test_grounded_case_fails_when_expected_refused_but_chunks_found():
    case = _case(expected_type="refused", expected_sources=())
    chunks = [_chunk("a", "/data/report.txt")]
    result = score_case(case, chunks, "a real answer", 0.05, 1.2)
    assert result.type_match is False
    assert result.passed is False


def test_keyword_match_is_informational_and_does_not_affect_passed():
    case = _case(expected_sources=("report.txt",), expected_keywords=("nonexistent-word",))
    chunks = [_chunk("a", "/data/report.txt")]
    result = score_case(case, chunks, "a real answer", 0.05, 1.2)
    assert result.keyword_match is False
    assert result.type_match is True
    assert result.sources_match is True
    assert result.passed is True  # keyword mismatch alone must not fail the case


def test_keyword_match_is_none_when_no_keywords_declared():
    case = _case(expected_sources=("report.txt",), expected_keywords=())
    chunks = [_chunk("a", "/data/report.txt")]
    result = score_case(case, chunks, "anything", 0.05, 1.2)
    assert result.keyword_match is None


# ── Refusal-result scoring ─────────────────────────────────────────────────────

def test_refused_case_passes_when_zero_chunks_and_no_expected_sources():
    case = _case(expected_type="refused", expected_sources=())
    result = score_case(
        case, [], "I could not find relevant information in the indexed documents.", 0.03, None
    )
    assert result.actual_type == "refused"
    assert result.type_match is True
    assert result.sources_match is True
    assert result.passed is True
    assert result.generation_time is None
    assert result.total_time == pytest.approx(0.03)


def test_refused_case_fails_when_chunks_unexpectedly_found():
    case = _case(expected_type="refused", expected_sources=())
    chunks = [_chunk("a", "/data/report.txt")]
    result = score_case(case, chunks, "some answer", 0.03, 1.0)
    assert result.actual_type == "grounded"
    assert result.type_match is False
    assert result.passed is False


# ── run_case / run_dataset — zero-retrieval cases skip the LLM ────────────────

def test_run_case_does_not_call_chat_model_when_retrieval_is_empty():
    case = _case(expected_type="refused", expected_sources=())
    retriever = MagicMock()
    retriever.retrieve.return_value = []
    runtime = MagicMock()

    result = run_case(case, retriever, runtime)

    runtime.stream_chat.assert_not_called()
    assert result.generation_time is None
    assert result.answer == "I could not find relevant information in the indexed documents."


def test_run_case_calls_chat_model_when_retrieval_is_non_empty():
    case = _case(expected_type="grounded", expected_sources=("report.txt",))
    retriever = MagicMock()
    retriever.retrieve.return_value = [_chunk("About RAG.", "/data/report.txt")]
    runtime = MagicMock()
    runtime.stream_chat.return_value = iter(["Grounded ", "answer."])

    result = run_case(case, retriever, runtime)

    runtime.stream_chat.assert_called_once()
    assert result.generation_time is not None
    assert result.answer == "Grounded answer."


def test_run_case_strips_think_block_from_generated_answer():
    """The evaluation path must never expose Qwen3 reasoning either."""
    case = _case(expected_type="grounded", expected_sources=("report.txt",))
    retriever = MagicMock()
    retriever.retrieve.return_value = [_chunk("About RAG.", "/data/report.txt")]
    runtime = MagicMock()
    runtime.stream_chat.return_value = iter(["<think>", "hidden reasoning", "</think>", "Visible."])

    result = run_case(case, retriever, runtime)

    assert result.answer == "Visible."
    assert "<think>" not in result.answer
    assert "hidden reasoning" not in result.answer


def test_run_dataset_runs_every_case_in_order():
    cases = [
        _case(id="c1", expected_type="refused", expected_sources=()),
        _case(id="c2", expected_type="refused", expected_sources=()),
    ]
    retriever = MagicMock()
    retriever.retrieve.return_value = []
    runtime = MagicMock()

    results = run_dataset(cases, retriever, runtime)

    assert [r.case.id for r in results] == ["c1", "c2"]
    runtime.stream_chat.assert_not_called()


# ── Deterministic summary counts ─────────────────────────────────────────────

def test_summarize_counts_are_deterministic():
    passing = _case(id="p", expected_type="refused", expected_sources=())
    failing = _case(id="f", expected_type="refused", expected_sources=())

    pass_result = score_case(passing, [], "refused", 0.01, None)
    chunks = [_chunk("a", "/data/report.txt")]
    fail_result = score_case(failing, chunks, "unexpected answer", 0.01, 1.0)

    summary = summarize([pass_result, fail_result])

    assert summary == {
        "total": 2,
        "passed": 1,
        "failed": 1,
        "type_mismatches": 1,
        "source_mismatches": 0,
        "keyword_mismatches": 0,
    }


def test_summarize_empty_list_is_all_zero():
    assert summarize([]) == {
        "total": 0,
        "passed": 0,
        "failed": 0,
        "type_mismatches": 0,
        "source_mismatches": 0,
        "keyword_mismatches": 0,
    }

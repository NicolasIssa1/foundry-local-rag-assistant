"""Schema and loader for the reusable M6 RAG evaluation dataset.

The dataset itself (src/evaluation/eval_dataset.json) is a plain,
human-readable JSON array of cases. Each case declares a question and the
objectively-checkable behaviour the pipeline is expected to produce for it
(grounded vs. refused, and — for grounded cases — which source document(s)
should appear among the retrieved chunks). expected_keywords is optional
and purely informational; see scoring.EvalResult.passed for why.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

_DATASET_PATH = Path(__file__).parent / "eval_dataset.json"

VALID_CATEGORIES = frozenset({"grounded", "grounded_paraphrase", "unrelated"})
VALID_EXPECTED_TYPES = frozenset({"grounded", "refused"})

_REQUIRED_FIELDS = ("id", "question", "category", "expected_type")


class InvalidEvalCaseError(ValueError):
    """Raised when an evaluation case is missing fields or has invalid values."""


@dataclass(frozen=True)
class EvalCase:
    id: str
    question: str
    category: str
    expected_type: str
    expected_sources: tuple[str, ...]
    expected_keywords: tuple[str, ...]


def _parse_case(raw: dict) -> EvalCase:
    if not isinstance(raw, dict):
        raise InvalidEvalCaseError(f"Eval case must be a JSON object, got: {raw!r}")

    missing = [f for f in _REQUIRED_FIELDS if f not in raw]
    if missing:
        raise InvalidEvalCaseError(f"Eval case missing required field(s) {missing}: {raw!r}")

    case_id = raw["id"]

    if not isinstance(raw["question"], str) or not raw["question"].strip():
        raise InvalidEvalCaseError(f"Eval case {case_id!r} has an empty/invalid 'question'")

    if raw["category"] not in VALID_CATEGORIES:
        raise InvalidEvalCaseError(
            f"Eval case {case_id!r} has invalid category {raw['category']!r}; "
            f"expected one of {sorted(VALID_CATEGORIES)}"
        )

    if raw["expected_type"] not in VALID_EXPECTED_TYPES:
        raise InvalidEvalCaseError(
            f"Eval case {case_id!r} has invalid expected_type {raw['expected_type']!r}; "
            f"expected one of {sorted(VALID_EXPECTED_TYPES)}"
        )

    expected_sources = tuple(raw.get("expected_sources", []))
    expected_keywords = tuple(raw.get("expected_keywords", []))

    if raw["expected_type"] == "refused" and expected_sources:
        raise InvalidEvalCaseError(
            f"Eval case {case_id!r} is expected_type='refused' but declares "
            f"expected_sources={expected_sources!r}; refused cases must have none"
        )
    if raw["expected_type"] == "grounded" and not expected_sources:
        raise InvalidEvalCaseError(
            f"Eval case {case_id!r} is expected_type='grounded' but declares "
            f"no expected_sources"
        )

    return EvalCase(
        id=case_id,
        question=raw["question"],
        category=raw["category"],
        expected_type=raw["expected_type"],
        expected_sources=expected_sources,
        expected_keywords=expected_keywords,
    )


def load_dataset(path: str | Path | None = None) -> list[EvalCase]:
    """Load and validate the evaluation dataset (a JSON array of cases).

    Defaults to the bundled src/evaluation/eval_dataset.json; pass `path`
    to load a different dataset file (e.g. in tests).
    """
    dataset_path = Path(path) if path is not None else _DATASET_PATH
    raw_cases = json.loads(dataset_path.read_text(encoding="utf-8"))

    if not isinstance(raw_cases, list) or not raw_cases:
        raise InvalidEvalCaseError(f"Dataset at {dataset_path} must be a non-empty JSON array")

    cases = [_parse_case(raw) for raw in raw_cases]

    ids = [c.id for c in cases]
    duplicates = sorted({i for i in ids if ids.count(i) > 1})
    if duplicates:
        raise InvalidEvalCaseError(f"Duplicate eval case id(s): {duplicates}")

    return cases

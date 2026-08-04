"""Tests for the deterministic answer-finalisation pass —
src/llm/answer_finalizer.py.
"""
from src.llm.answer_finalizer import finalize_answer

LONG_SENTENCE = "This is a sufficiently long sentence to count as repeated content."


# ── Already-clean input is left untouched ────────────────────────────────────

def test_short_answer_below_limit_returned_unchanged():
    text = "RAG combines retrieval with generation to ground the model's answer."
    assert finalize_answer(text) == text


def test_single_word_answer_unchanged():
    assert finalize_answer("hello") == "hello"


def test_multi_paragraph_non_duplicate_answer_unchanged():
    text = "First paragraph explains RAG.\n\nSecond paragraph explains chunking."
    assert finalize_answer(text) == text


# ── Think-tag stripping ───────────────────────────────────────────────────────

def test_strips_complete_think_block():
    out = finalize_answer("<think>internal reasoning</think>Visible answer.")
    assert out == "Visible answer."
    assert "<think>" not in out
    assert "internal reasoning" not in out


def test_strips_stray_unterminated_think_tag():
    out = finalize_answer("<think>Visible answer.")
    assert "<think>" not in out
    assert "Visible answer." in out


def test_strips_stray_closing_think_tag():
    out = finalize_answer("Visible answer.</think>")
    assert "</think>" not in out
    assert "Visible answer." in out


# ── Whitespace normalization ──────────────────────────────────────────────────

def test_collapses_repeated_horizontal_whitespace():
    out = finalize_answer("Too    many   spaces.")
    assert out == "Too many spaces."


def test_collapses_excess_blank_lines():
    out = finalize_answer("Para one.\n\n\n\n\nPara two.")
    assert out == "Para one.\n\nPara two."


def test_strips_leading_and_trailing_whitespace():
    out = finalize_answer("   Answer with padding.   ")
    assert out == "Answer with padding."


def test_preserves_single_newline_list_structure():
    text = "1. First item.\n2. Second item.\n3. Third item."
    assert finalize_answer(text) == text


# ── Duplicate paragraph removal ───────────────────────────────────────────────

def test_removes_exact_duplicate_paragraph():
    text = "RAG grounds answers in retrieved context.\n\nRAG grounds answers in retrieved context."
    out = finalize_answer(text)
    assert out == "RAG grounds answers in retrieved context."


def test_removes_duplicate_paragraph_case_and_whitespace_insensitive():
    text = "RAG grounds answers in context.\n\nRAG   GROUNDS answers in context."
    out = finalize_answer(text)
    assert out.count("grounds") == 1 or out.lower().count("grounds") == 1


def test_keeps_distinct_paragraphs():
    text = "First distinct point.\n\nSecond distinct point."
    assert finalize_answer(text) == text


# ── Duplicate sentence removal ────────────────────────────────────────────────

def test_removes_repeated_sentence_within_a_paragraph():
    text = f"{LONG_SENTENCE} Something else entirely different. {LONG_SENTENCE}"
    out = finalize_answer(text)
    assert out.count(LONG_SENTENCE) == 1


def test_keeps_short_repeated_phrases_below_word_threshold():
    """Repetition detection only targets substantive (>= 6 word) repeats —
    short incidental repeats (e.g. 'RAG' used twice) must survive."""
    text = "RAG is useful. RAG stands for Retrieval-Augmented Generation."
    out = finalize_answer(text)
    assert out.lower().count("rag") >= 2


def test_removes_repeated_sentence_across_paragraph_boundary():
    text = f"{LONG_SENTENCE}\n\nA distinct middle point.\n\n{LONG_SENTENCE}"
    out = finalize_answer(text)
    assert out.count(LONG_SENTENCE) == 1


# ── Word-limit enforcement ────────────────────────────────────────────────────

def test_answer_already_under_limit_is_unchanged():
    text = " ".join(["word"] * 50) + "."
    assert finalize_answer(text, max_words=150) == text


def test_truncates_at_last_complete_sentence_within_limit():
    first = "Short first sentence here."
    second = " ".join(["padding"] * 20) + "."
    third = "This sentence pushes the total over the limit with more padding words here."
    text = f"{first} {second} {third}"
    out = finalize_answer(text, max_words=25)
    assert out == f"{first} {second}"
    assert out.endswith(".")
    assert len(out.split()) <= 25


def test_hard_cutoff_when_first_sentence_alone_exceeds_limit():
    text = " ".join(["word"] * 40) + "."
    out = finalize_answer(text, max_words=10)
    assert len(out.split()) == 10


def test_truncation_never_fabricates_new_words():
    first = "Alpha beta gamma delta."
    second = "Epsilon zeta eta theta iota kappa lambda mu nu xi omicron pi rho sigma."
    text = f"{first} {second}"
    out = finalize_answer(text, max_words=6)
    for word in out.replace(".", "").split():
        assert word in text

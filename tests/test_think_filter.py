"""
Tests for the streaming <think>...</think> suppression filter —
src/llm/think_filter.py.

Chunks are fed in various boundary configurations to mirror how a real
token stream might split the open/close tags arbitrarily.
"""
from src.llm.think_filter import ThinkBlockFilter, filter_think_stream


def _run(chunks: list[str]) -> str:
    return "".join(filter_think_stream(iter(chunks)))


# ── 1. Complete think block followed by an answer ────────────────────────────

def test_complete_think_block_in_one_chunk_then_answer():
    out = _run(["<think>internal reasoning</think>Final answer."])
    assert out == "Final answer."
    assert "internal reasoning" not in out


def test_complete_think_block_and_answer_in_separate_chunks():
    out = _run(["<think>internal reasoning</think>", "Final answer."])
    assert out == "Final answer."


# ── 2. Opening tag split across chunks ────────────────────────────────────────

def test_opening_tag_split_across_two_chunks():
    out = _run(["<thi", "nk>reasoning</think>Answer"])
    assert out == "Answer"


def test_opening_tag_split_character_by_character():
    chunks = list("<think>") + ["reasoning", "</think>", "Answer"]
    out = _run(chunks)
    assert out == "Answer"


# ── 3. Closing tag split across chunks ────────────────────────────────────────

def test_closing_tag_split_across_two_chunks():
    out = _run(["<think>reasoning</th", "ink>Answer"])
    assert out == "Answer"


def test_closing_tag_split_character_by_character():
    chunks = ["<think>reasoning"] + list("</think>") + ["Answer"]
    out = _run(chunks)
    assert out == "Answer"


# ── 4. Closing tag and answer text in the same chunk ──────────────────────────

def test_closing_tag_and_answer_in_same_chunk():
    out = _run(["<think>reasoning", "</think>Answer text here"])
    assert out == "Answer text here"


# ── 5. No think block ─────────────────────────────────────────────────────────

def test_no_think_block_passes_through_unchanged():
    out = _run(["Just a normal, grounded answer."])
    assert out == "Just a normal, grounded answer."


def test_no_think_block_split_across_many_chunks():
    out = _run(["Just ", "a normal, ", "grounded answer."])
    assert out == "Just a normal, grounded answer."


# ── 6. Empty think block ──────────────────────────────────────────────────────

def test_empty_think_block_in_one_chunk():
    out = _run(["<think></think>Answer"])
    assert out == "Answer"


def test_empty_think_block_split_across_chunks():
    out = _run(["<think>", "</think>", "Answer"])
    assert out == "Answer"


# ── 7. Incomplete think block at end of stream ────────────────────────────────

def test_incomplete_think_block_at_end_of_stream_is_discarded():
    out = _run(["<think>reasoning that never closes because generation stopped"])
    assert out == ""


def test_incomplete_think_block_does_not_swallow_preceding_visible_text():
    out = _run(["Before.", "<think>unterminated reasoning</th"])
    assert out == "Before."


def test_incomplete_open_tag_at_end_of_stream_is_flushed_as_literal_text():
    """If the stream ends before "<think>" is ever completed, the held-back
    partial prefix was never actually a tag and must be released as-is."""
    out = _run(["Rate is <thi"])
    assert out == "Rate is <thi"


# ── 8. Ordinary visible text containing "think" ───────────────────────────────

def test_ordinary_text_containing_the_word_think_is_preserved():
    out = _run(["I think this design is correct."])
    assert out == "I think this design is correct."


def test_ordinary_text_with_think_split_across_chunk_boundary():
    out = _run(["I th", "ink this works."])
    assert out == "I think this works."


def test_angle_bracket_not_mistaken_for_tag_start_when_not_followed_by_think():
    out = _run(["Value is < 5 and that's fine."])
    assert out == "Value is < 5 and that's fine."


def test_angle_bracket_split_at_chunk_boundary_still_resolves_correctly():
    out = _run(["Value is <", " 5 and that's fine."])
    assert out == "Value is < 5 and that's fine."


# ── Extra robustness ───────────────────────────────────────────────────────────

def test_multiple_think_blocks_are_all_suppressed():
    out = _run(["<think>a</think>Part 1. <think>b</think>Part 2."])
    assert out == "Part 1. Part 2."


def test_class_level_feed_and_flush_api_directly():
    filt = ThinkBlockFilter()
    assert filt.feed("<think>reasoning") == ""
    assert filt.feed("</think>Answer") == "Answer"
    assert filt.flush() == ""

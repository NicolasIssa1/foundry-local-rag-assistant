"""Deterministic, reusable finalisation pass applied to every generated
answer before it is shown or scored — independent of any particular
question, corpus, or evaluation case.

Applies, in order:
  1. Strip any literal <think>...</think> content that reached this far
     (defence-in-depth; the streaming ThinkBlockFilter should already have
     removed it).
  2. Normalise incidental whitespace (repeated spaces/tabs, 3+ blank
     lines) without touching intentional single-newline structure (e.g.
     numbered lists).
  3. Drop duplicate paragraphs (verbatim, whitespace/case-insensitive).
  4. Drop duplicate sentences (>= MIN_REPEATED_WORDS words, mirroring
     evaluation.safety_checks.find_repeated_blocks) — keeps the first
     occurrence, removes the rest, preserving original separators around
     everything that wasn't a duplicate.
  5. Enforce a maximum word count, truncating at the last complete
     sentence boundary within the limit when possible, falling back to a
     hard word-count cutoff only if no sentence fits.

Callers apply this only to the raw model answer text; the Sources section
is built and appended separately and is never passed through here.
"""
from __future__ import annotations

import re

DEFAULT_MAX_WORDS = 150
MIN_REPEATED_WORDS = 6

_THINK_BLOCK = re.compile(r"<think>.*?</think>", re.IGNORECASE | re.DOTALL)
_STRAY_THINK_TAG = re.compile(r"</?think>", re.IGNORECASE)
_SENTENCE_END = re.compile(r"[.!?](?=\s|$)")
_SENTENCE_SPLIT_KEEP_SEP = re.compile(r"(?<=[.!?])(\s+)")


def _strip_think_tags(text: str) -> str:
    text = _THINK_BLOCK.sub("", text)
    return _STRAY_THINK_TAG.sub("", text)


def _normalize_whitespace(text: str) -> str:
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    lines = [re.sub(r"[ \t]+", " ", line).strip() for line in text.split("\n")]
    text = "\n".join(lines)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def _paragraphs(text: str) -> list[str]:
    return [p for p in re.split(r"\n\s*\n", text) if p.strip()]


def _remove_duplicate_paragraphs(text: str) -> str:
    seen: set[str] = set()
    kept: list[str] = []
    for para in _paragraphs(text):
        norm = re.sub(r"\s+", " ", para.strip().lower())
        if norm in seen:
            continue
        seen.add(norm)
        kept.append(para)
    return "\n\n".join(kept)


def _remove_duplicate_sentences(text: str) -> str:
    """Drop repeated sentences (>= MIN_REPEATED_WORDS words), keeping the
    original separators (spaces/newlines) around everything that wasn't a
    duplicate so list/paragraph structure survives untouched."""
    seen: set[str] = set()
    tokens = _SENTENCE_SPLIT_KEEP_SEP.split(text)  # [sent, sep, sent, sep, ..., sent]
    kept: list[str] = []
    skip_next_separator = False

    for i, tok in enumerate(tokens):
        is_separator = i % 2 == 1
        if is_separator:
            if skip_next_separator:
                skip_next_separator = False
                continue
            kept.append(tok)
            continue

        norm = re.sub(r"\s+", " ", tok.strip().lower())
        if len(norm.split()) >= MIN_REPEATED_WORDS:
            if norm in seen:
                skip_next_separator = True
                continue
            seen.add(norm)
        kept.append(tok)

    return "".join(kept)


def _truncate_to_word_limit(text: str, max_words: int) -> str:
    words = text.split()
    if len(words) <= max_words:
        return text

    best: str | None = None
    for match in _SENTENCE_END.finditer(text):
        candidate = text[: match.end()].strip()
        if len(candidate.split()) <= max_words:
            best = candidate
        else:
            break

    if best:
        return best

    return " ".join(words[:max_words])


def finalize_answer(raw_answer: str, max_words: int = DEFAULT_MAX_WORDS) -> str:
    """Deterministically finalise a raw model answer for display/scoring."""
    text = _strip_think_tags(raw_answer)
    text = _normalize_whitespace(text)
    text = _remove_duplicate_paragraphs(text)
    text = _remove_duplicate_sentences(text)
    text = _normalize_whitespace(text)
    text = _truncate_to_word_limit(text, max_words)
    return text

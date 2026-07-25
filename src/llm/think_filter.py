"""Streaming filter that suppresses <think>...</think> reasoning blocks.

Some chat models (e.g. Qwen3 in reasoning mode) prepend their internal
chain-of-thought to every response, wrapped in a literal
"<think>...</think>" block, before the actual answer. This module strips
that block from a live token stream without buffering the whole response
first, so it works with the CLI's token-by-token printing.

The open/close tags may be split arbitrarily across stream chunks (e.g. one
chunk ends mid-tag), so a naive per-chunk regex would either leak partial
tags or drop visible text. ThinkBlockFilter instead tracks a small amount of
state across feed() calls: whether it is currently inside a think block, and
a buffer holding only text that might still turn out to be part of a tag.
"""
from __future__ import annotations

from typing import Iterable, Iterator

_OPEN_TAG = "<think>"
_CLOSE_TAG = "</think>"


class ThinkBlockFilter:
    """Stateful, chunk-boundary-safe filter for a single response stream.

    Usage:
        filt = ThinkBlockFilter()
        for chunk in raw_stream:
            visible = filt.feed(chunk)
            if visible:
                ...emit visible...
        trailing = filt.flush()  # call once, after the stream ends
        if trailing:
            ...emit trailing...
    """

    def __init__(self) -> None:
        self._buffer = ""
        self._inside_think = False

    def feed(self, chunk: str) -> str:
        """Feed the next raw chunk; returns the visible text safe to emit now."""
        self._buffer += chunk
        visible: list[str] = []

        while True:
            if not self._inside_think:
                idx = self._buffer.find(_OPEN_TAG)
                if idx == -1:
                    safe = self._safe_prefix_length(self._buffer, _OPEN_TAG)
                    visible.append(self._buffer[:safe])
                    self._buffer = self._buffer[safe:]
                    break
                visible.append(self._buffer[:idx])
                self._buffer = self._buffer[idx + len(_OPEN_TAG):]
                self._inside_think = True
            else:
                idx = self._buffer.find(_CLOSE_TAG)
                if idx == -1:
                    # Everything up to the safe boundary is reasoning text —
                    # discard it (never emitted). Keep only a tail that could
                    # still grow into "</think>" once more text arrives.
                    safe = self._safe_prefix_length(self._buffer, _CLOSE_TAG)
                    self._buffer = self._buffer[safe:]
                    break
                self._buffer = self._buffer[idx + len(_CLOSE_TAG):]
                self._inside_think = False

        return "".join(visible)

    def flush(self) -> str:
        """Call once after the stream ends; releases any trailing buffered text.

        If the stream ended while still inside an (unterminated) think block,
        the buffered reasoning text is discarded rather than leaked.
        """
        remaining = self._buffer
        self._buffer = ""
        if self._inside_think:
            return ""
        return remaining

    @staticmethod
    def _safe_prefix_length(buf: str, tag: str) -> int:
        """Length of `buf` guaranteed not to be the start of `tag`.

        The rest (a suffix of at most len(tag) - 1 chars) might still grow
        into `tag` once the next chunk arrives, so it must stay buffered
        rather than be emitted/discarded yet.
        """
        max_suffix = min(len(tag) - 1, len(buf))
        for size in range(max_suffix, 0, -1):
            if tag.startswith(buf[-size:]):
                return len(buf) - size
        return len(buf)


def filter_think_stream(chunks: Iterable[str]) -> Iterator[str]:
    """Wrap a raw token stream, suppressing <think>...</think> blocks live."""
    filt = ThinkBlockFilter()
    for chunk in chunks:
        visible = filt.feed(chunk)
        if visible:
            yield visible
    trailing = filt.flush()
    if trailing:
        yield trailing

from __future__ import annotations

from .models import Chunk, Document

# Defaults chosen so each chunk fits comfortably inside a 512-token
# embedding model (1 token ≈ 4 chars → 1000 chars ≈ 250 tokens).
DEFAULT_CHUNK_SIZE = 1000
DEFAULT_OVERLAP = 200


def _snap_to_word_boundary(text: str, pos: int, search_backwards: bool) -> int:
    """Move pos to the nearest space so we never cut a word in half."""
    if pos <= 0 or pos >= len(text):
        return pos
    if text[pos] == " ":
        return pos
    if search_backwards:
        boundary = text.rfind(" ", 0, pos)
        return boundary if boundary != -1 else pos
    else:
        boundary = text.find(" ", pos)
        return boundary if boundary != -1 else pos


def chunk(
    doc: Document,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    overlap: int = DEFAULT_OVERLAP,
) -> list[Chunk]:
    """Split a cleaned Document into overlapping Chunk objects.

    Each chunk is at most chunk_size characters wide. Consecutive chunks
    share overlap characters so that sentences at boundaries appear
    complete in at least one chunk. End positions are snapped to the
    nearest word boundary to avoid splitting words.

    Raises ValueError if overlap >= chunk_size (would produce infinite loop).
    """
    if overlap >= chunk_size:
        raise ValueError(
            f"overlap ({overlap}) must be less than chunk_size ({chunk_size})"
        )

    text = doc.content
    if not text.strip():
        return []

    step = chunk_size - overlap
    chunks: list[Chunk] = []
    chunk_index = 0
    start = 0

    while start < len(text):
        raw_end = min(start + chunk_size, len(text))

        # Snap end backwards to a word boundary (only when not at EOF)
        end = (
            _snap_to_word_boundary(text, raw_end, search_backwards=True)
            if raw_end < len(text)
            else raw_end
        )
        # Guard: snapping must not collapse the chunk to nothing
        if end <= start:
            end = raw_end

        chunk_text = text[start:end].strip()

        if chunk_text:
            chunks.append(
                Chunk(
                    text=chunk_text,
                    source=doc.source,
                    file_type=doc.file_type,
                    page=doc.page,
                    chunk_index=chunk_index,
                    start_char=start,
                    end_char=end,
                    metadata=doc.metadata,
                )
            )
            chunk_index += 1

        next_start = start + step
        if next_start >= len(text):
            break
        start = next_start

    return chunks

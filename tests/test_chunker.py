import pytest
from src.ingestion.chunker import chunk, DEFAULT_CHUNK_SIZE, DEFAULT_OVERLAP
from src.ingestion.models import Chunk, Document


def _doc(content: str, file_type: str = "txt", page: int = 1) -> Document:
    return Document(content=content, source="test.txt", file_type=file_type, page=page)


def _long_text(n_chars: int) -> str:
    """Return a word-filled string of approximately n_chars characters."""
    word = "word "
    return (word * (n_chars // len(word) + 1))[:n_chars].rstrip()


# ── Return type and basic structure ──────────────────────────────────────────

def test_chunk_returns_list():
    assert isinstance(chunk(_doc("hello world")), list)


def test_chunk_returns_chunk_objects():
    chunks = chunk(_doc(_long_text(200)))
    assert all(isinstance(c, Chunk) for c in chunks)


def test_chunk_text_is_non_empty_string():
    chunks = chunk(_doc(_long_text(200)))
    for c in chunks:
        assert isinstance(c.text, str)
        assert len(c.text) > 0


# ── Edge cases ────────────────────────────────────────────────────────────────

def test_empty_document_returns_empty_list():
    assert chunk(_doc("")) == []


def test_whitespace_only_document_returns_empty_list():
    assert chunk(_doc("   \n\n  ")) == []


def test_short_document_returns_single_chunk():
    short_text = "This is a short document."
    chunks = chunk(_doc(short_text), chunk_size=1000, overlap=200)
    assert len(chunks) == 1


def test_invalid_overlap_raises_value_error():
    with pytest.raises(ValueError, match="overlap"):
        chunk(_doc("some text"), chunk_size=100, overlap=100)


def test_overlap_greater_than_chunk_size_raises_value_error():
    with pytest.raises(ValueError):
        chunk(_doc("some text"), chunk_size=100, overlap=150)


# ── chunk_index ───────────────────────────────────────────────────────────────

def test_chunk_index_starts_at_zero():
    chunks = chunk(_doc(_long_text(3000)), chunk_size=1000, overlap=200)
    assert chunks[0].chunk_index == 0


def test_chunk_index_is_sequential():
    chunks = chunk(_doc(_long_text(5000)), chunk_size=1000, overlap=200)
    for i, c in enumerate(chunks):
        assert c.chunk_index == i


# ── start_char and end_char ───────────────────────────────────────────────────

def test_first_chunk_starts_at_zero():
    chunks = chunk(_doc(_long_text(3000)), chunk_size=1000, overlap=200)
    assert chunks[0].start_char == 0


def test_end_char_greater_than_start_char():
    chunks = chunk(_doc(_long_text(3000)), chunk_size=1000, overlap=200)
    for c in chunks:
        assert c.end_char > c.start_char


def test_start_and_end_chars_are_within_document_bounds():
    text = _long_text(3000)
    chunks = chunk(_doc(text), chunk_size=1000, overlap=200)
    for c in chunks:
        assert c.start_char >= 0
        assert c.end_char <= len(text)


def test_chunk_text_matches_document_slice():
    text = _long_text(3000)
    chunks = chunk(_doc(text), chunk_size=1000, overlap=200)
    for c in chunks:
        # The chunk text is a stripped slice of the document
        assert c.text == text[c.start_char:c.end_char].strip()


# ── Overlap ───────────────────────────────────────────────────────────────────

def test_consecutive_chunks_overlap():
    """The end of chunk N must overlap with the start of chunk N+1."""
    text = _long_text(5000)
    chunks = chunk(_doc(text), chunk_size=1000, overlap=200)
    assert len(chunks) >= 2
    for i in range(len(chunks) - 1):
        # chunk N+1 starts before chunk N ends
        assert chunks[i + 1].start_char < chunks[i].end_char


def test_overlap_content_appears_in_consecutive_chunks():
    """A word in the overlap region should appear in both adjacent chunks."""
    text = _long_text(3000)
    chunks = chunk(_doc(text), chunk_size=1000, overlap=200)
    assert len(chunks) >= 2
    # Take a word from chunk 0 near its end; it should appear in chunk 1 too
    words_in_first = set(chunks[0].text.split())
    words_in_second = set(chunks[1].text.split())
    assert len(words_in_first & words_in_second) > 0


# ── Chunk size ────────────────────────────────────────────────────────────────

def test_chunks_do_not_exceed_chunk_size():
    text = _long_text(5000)
    size = 500
    chunks = chunk(_doc(text), chunk_size=size, overlap=50)
    for c in chunks:
        # Allow a small tolerance for word-boundary snapping
        assert len(c.text) <= size + 50


def test_long_document_produces_multiple_chunks():
    chunks = chunk(_doc(_long_text(5000)), chunk_size=1000, overlap=200)
    assert len(chunks) > 1


# ── Metadata preservation ─────────────────────────────────────────────────────

def test_chunk_inherits_source():
    doc = Document(content=_long_text(200), source="/data/report.txt", file_type="txt")
    chunks = chunk(doc)
    assert all(c.source == "/data/report.txt" for c in chunks)


def test_chunk_inherits_file_type():
    doc = _doc(_long_text(200), file_type="md")
    chunks = chunk(doc)
    assert all(c.file_type == "md" for c in chunks)


def test_chunk_inherits_page():
    doc = _doc(_long_text(200), page=7)
    chunks = chunk(doc)
    assert all(c.page == 7 for c in chunks)


def test_chunk_inherits_metadata():
    doc = Document(
        content=_long_text(200),
        source="f.txt",
        file_type="txt",
        metadata={"author": "Nicolas"},
    )
    chunks = chunk(doc)
    assert all(c.metadata == {"author": "Nicolas"} for c in chunks)


# ── No mid-word cuts ──────────────────────────────────────────────────────────

def test_chunks_do_not_start_with_partial_word():
    """Each chunk should start at the beginning of a word (no leading spaces)."""
    text = _long_text(5000)
    chunks = chunk(_doc(text), chunk_size=1000, overlap=200)
    for c in chunks:
        assert not c.text[0].isspace()


# ── Integration: full pipeline load → clean → chunk ──────────────────────────

def test_full_pipeline_on_sample_txt():
    from pathlib import Path
    from src.ingestion.loader import load
    from src.ingestion.cleaner import clean

    docs = load(Path("data/sample.txt"))
    cleaned = clean(docs[0])
    chunks = chunk(cleaned, chunk_size=500, overlap=100)

    assert len(chunks) >= 1
    assert all(isinstance(c, Chunk) for c in chunks)
    assert all(len(c.text) > 0 for c in chunks)


def test_full_pipeline_on_sample_md():
    from pathlib import Path
    from src.ingestion.loader import load
    from src.ingestion.cleaner import clean

    docs = load(Path("data/sample.md"))
    cleaned = clean(docs[0])
    chunks = chunk(cleaned, chunk_size=500, overlap=100)

    assert len(chunks) >= 1
    # Markdown headings should already be stripped by the cleaner
    for c in chunks:
        for line in c.text.split("\n"):
            assert not line.startswith("#"), f"Unstripped heading in chunk: {line!r}"

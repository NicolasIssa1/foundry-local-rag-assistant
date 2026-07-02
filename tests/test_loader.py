import pytest
from pathlib import Path

from src.ingestion import load, Document
from src.ingestion.loader import load_file, SUPPORTED_EXTENSIONS

DATA_DIR = Path(__file__).parent.parent / "data"


# ── Happy-path tests ──────────────────────────────────────────────────────────

def test_load_txt_file_returns_single_document():
    docs = load(DATA_DIR / "sample.txt")
    assert len(docs) == 1


def test_load_txt_file_type():
    docs = load(DATA_DIR / "sample.txt")
    assert docs[0].file_type == "txt"


def test_load_txt_content_is_non_empty():
    docs = load(DATA_DIR / "sample.txt")
    assert len(docs[0].content) > 0


def test_load_txt_source_contains_filename():
    docs = load(DATA_DIR / "sample.txt")
    assert "sample.txt" in docs[0].source


def test_load_md_file_returns_single_document():
    docs = load(DATA_DIR / "sample.md")
    assert len(docs) == 1


def test_load_md_file_type():
    docs = load(DATA_DIR / "sample.md")
    assert docs[0].file_type == "md"


def test_load_md_content_is_non_empty():
    docs = load(DATA_DIR / "sample.md")
    assert len(docs[0].content) > 0


def test_load_directory_returns_multiple_documents():
    docs = load(DATA_DIR)
    assert len(docs) >= 2


def test_load_directory_includes_both_file_types():
    docs = load(DATA_DIR)
    file_types = {doc.file_type for doc in docs}
    assert "txt" in file_types
    assert "md" in file_types


def test_document_page_defaults_to_one():
    docs = load(DATA_DIR / "sample.txt")
    assert docs[0].page == 1


def test_document_metadata_defaults_to_empty_dict():
    docs = load(DATA_DIR / "sample.txt")
    assert docs[0].metadata == {}


def test_document_source_is_absolute_path():
    docs = load(DATA_DIR / "sample.txt")
    assert Path(docs[0].source).is_absolute()


def test_directory_results_are_sorted():
    docs = load(DATA_DIR)
    sources = [doc.source for doc in docs]
    assert sources == sorted(sources)


# ── Document dataclass contract ───────────────────────────────────────────────

def test_document_is_correct_type():
    docs = load(DATA_DIR / "sample.txt")
    assert isinstance(docs[0], Document)


def test_document_content_is_str():
    docs = load(DATA_DIR / "sample.txt")
    assert isinstance(docs[0].content, str)


# ── Error-path tests ──────────────────────────────────────────────────────────

def test_load_nonexistent_path_raises_file_not_found():
    with pytest.raises(FileNotFoundError):
        load("nonexistent_file.txt")


def test_load_unsupported_extension_raises_value_error(tmp_path):
    bad_file = tmp_path / "document.pdf"
    bad_file.write_text("placeholder")
    with pytest.raises(ValueError, match="Unsupported file type"):
        load_file(bad_file)


def test_supported_extensions_constant():
    assert ".txt" in SUPPORTED_EXTENSIONS
    assert ".md" in SUPPORTED_EXTENSIONS

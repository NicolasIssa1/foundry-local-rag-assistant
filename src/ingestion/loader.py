from __future__ import annotations

from pathlib import Path

from .models import Document

SUPPORTED_EXTENSIONS: frozenset[str] = frozenset({".txt", ".md"})


def _read_text(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        # Fallback for files saved with a legacy encoding
        return path.read_text(encoding="latin-1")


def _load_txt(path: Path) -> Document:
    return Document(
        content=_read_text(path),
        source=str(path.resolve()),
        file_type="txt",
    )


def _load_md(path: Path) -> Document:
    return Document(
        content=_read_text(path),
        source=str(path.resolve()),
        file_type="md",
    )


_LOADERS = {
    ".txt": _load_txt,
    ".md": _load_md,
}


def load_file(path: Path) -> Document:
    """Load a single supported file and return a Document."""
    ext = path.suffix.lower()
    loader_fn = _LOADERS.get(ext)
    if loader_fn is None:
        raise ValueError(
            f"Unsupported file type: {ext!r}. "
            f"Supported extensions: {sorted(SUPPORTED_EXTENSIONS)}"
        )
    return loader_fn(path)


def load(path: str | Path) -> list[Document]:
    """Load one file or all supported files under a directory (recursive).

    Returns documents sorted by source path for deterministic ordering.
    """
    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(f"Path does not exist: {path}")

    if path.is_file():
        return [load_file(path)]

    documents: list[Document] = []
    for file_path in sorted(path.rglob("*")):
        if file_path.is_file() and file_path.suffix.lower() in SUPPORTED_EXTENSIONS:
            documents.append(load_file(file_path))

    return documents

from __future__ import annotations

import re

from .models import Document

# Markdown patterns applied in order — order matters
_MD_RULES: list[tuple[str, str]] = [
    # Images — remove entirely (no text value)
    (r"!\[.*?\]\(.*?\)", ""),
    # Links — keep the display text
    (r"\[([^\]]+)\]\([^\)]+\)", r"\1"),
    # Fenced code block delimiters (``` or ~~~) — remove the fence lines
    (r"^```[^\n]*\n?", "", ),   # handled below via re.MULTILINE
    # Headings — strip leading # characters
    (r"^#{1,6}\s+", ""),
    # Bold + italic combined (***text***)
    (r"\*{3}([^*]+)\*{3}", r"\1"),
    # Bold (**text** or __text__)
    (r"\*{2}([^*]+)\*{2}", r"\1"),
    (r"__([^_]+)__", r"\1"),
    # Italic (*text* or _text_)
    (r"\*([^*]+)\*", r"\1"),
    (r"_([^_]+)_", r"\1"),
    # Inline code (`code`)
    (r"`([^`]+)`", r"\1"),
    # Table separator rows (|---|---|)
    (r"^\|[\s\-\|:]+\|$", ""),
    # Horizontal rules
    (r"^[-*=]{3,}\s*$", ""),
]

# Pre-compiled with MULTILINE so ^ and $ match line boundaries
_MD_COMPILED = [
    (re.compile(pattern, re.MULTILINE), replacement)
    for pattern, replacement in _MD_RULES
]

# Separate rule for fenced code blocks (needs DOTALL to span lines)
_FENCED_CODE_BLOCK = re.compile(r"```.*?```", re.DOTALL)


def _clean_common(text: str) -> str:
    """Normalise whitespace and line endings for any file type."""
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    lines = [line.rstrip() for line in text.split("\n")]
    # Collapse 3+ consecutive blank lines into one
    cleaned: list[str] = []
    blank_count = 0
    for line in lines:
        if line == "":
            blank_count += 1
            if blank_count <= 1:
                cleaned.append(line)
        else:
            blank_count = 0
            cleaned.append(line)
    return "\n".join(cleaned).strip()


def _clean_markdown(text: str) -> str:
    """Strip Markdown syntax while preserving all readable content."""
    # Remove fenced code blocks' delimiters but keep the code body
    text = _FENCED_CODE_BLOCK.sub(
        lambda m: "\n".join(
            line for line in m.group(0).split("\n")
            if not line.startswith("```")
        ),
        text,
    )
    for pattern, replacement in _MD_COMPILED:
        text = pattern.sub(replacement, text)
    return text


def clean(doc: Document) -> Document:
    """Return a new Document with normalised, markup-free content."""
    text = doc.content

    if doc.file_type == "md":
        text = _clean_markdown(text)

    text = _clean_common(text)

    return Document(
        content=text,
        source=doc.source,
        file_type=doc.file_type,
        page=doc.page,
        metadata=doc.metadata,
    )

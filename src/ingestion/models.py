from dataclasses import dataclass, field


@dataclass
class Document:
    content: str
    source: str
    file_type: str
    page: int = 1
    metadata: dict = field(default_factory=dict)


@dataclass
class Chunk:
    text: str
    source: str
    file_type: str
    page: int
    chunk_index: int
    start_char: int
    end_char: int
    metadata: dict = field(default_factory=dict)

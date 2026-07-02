from dataclasses import dataclass, field


@dataclass
class Document:
    content: str
    source: str
    file_type: str
    page: int = 1
    metadata: dict = field(default_factory=dict)

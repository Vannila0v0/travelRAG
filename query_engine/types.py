from dataclasses import dataclass, field
from typing import Any


@dataclass
class Source:
    doc_id: str | None = None
    chunk_id: str | None = None
    source_path: str | None = None
    file_name: str | None = None
    chunk_index: int | None = None
    page: int | None = None
    section: str | None = None
    text: str | None = None
    score: float | None = None


@dataclass
class QueryResult:
    answer: str
    route: str
    sources: list[Source] = field(default_factory=list)
    contexts: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

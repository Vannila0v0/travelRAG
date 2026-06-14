from typing import Any, Literal

from pydantic import BaseModel, Field


RouteName = Literal["auto", "vector", "local", "global", "hybrid", "agent"]


class SourceModel(BaseModel):
    doc_id: str | None = None
    chunk_id: str | None = None
    source_path: str | None = None
    file_name: str | None = None
    chunk_index: int | None = None
    page: int | None = None
    section: str | None = None
    text: str | None = None
    score: float | None = None


class QueryRequest(BaseModel):
    question: str = Field(min_length=1)
    route: RouteName = "auto"
    include_source_text: bool = False
    max_sources: int = Field(default=5, ge=0, le=20)


class QueryResponse(BaseModel):
    route: str
    answer: str
    sources: list[SourceModel] = []
    metadata: dict[str, Any] = {}


class HealthResponse(BaseModel):
    status: Literal["ok", "degraded"]
    neo4j: bool
    faiss_index: bool
    details: dict[str, Any] = {}


class GraphStatsResponse(BaseModel):
    documents: int
    chunks: int
    entities: int
    entity_relationships: int
    mentions: int
    relationships_with_refs: int
    communities: int
    summarized_communities: int

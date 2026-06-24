from typing import Any, Literal

from pydantic import BaseModel, Field


RouteName = Literal["auto", "vector", "local", "global", "hybrid", "agent"]
ReportMode = Literal["concise", "full"]
PlanMode = Literal["auto", "detailed_itinerary", "place_recommendations"]
ResponseFormat = Literal["text", "itinerary"]
ErrorCode = Literal[
    "QUERY_FAILED",
    "VALIDATION_ERROR",
    "NEO4J_UNAVAILABLE",
    "FAISS_INDEX_MISSING",
    "LLM_CONFIG_MISSING",
    "EMBEDDING_CONFIG_MISSING",
    "ROUTE_NOT_AVAILABLE",
    "AGENT_EXECUTION_FAILED",
]


class SourceItem(BaseModel):
    doc_id: str | None = None
    chunk_id: str | None = None
    source_path: str | None = None
    file_name: str | None = None
    title: str | None = None
    url: str | None = None
    published_at: str | None = None
    source_type: str | None = None
    chunk_index: int | None = None
    page: int | None = None
    section: str | None = None
    text: str | None = None
    score: float | None = None


# Backward-compatible name used by existing routers/tests.
SourceModel = SourceItem


class ErrorDetail(BaseModel):
    code: ErrorCode | str
    message: str
    detail: str | None = None


class QueryMetadata(BaseModel):
    trace_id: str | None = None
    latency_ms: int | None = None
    requested_route: RouteName | str | None = None
    actual_route: str | None = None
    plan_mode: PlanMode | str | None = None
    response_format: ResponseFormat | str | None = None
    degraded: bool = False
    degraded_from: str | None = None
    degraded_to: str | None = None
    degradation_reason: str | None = None
    source_count: int = 0
    returned_source_count: int = 0
    engine_metadata: dict[str, Any] = Field(default_factory=dict)


class QueryRequest(BaseModel):
    question: str = Field(min_length=1)
    route: RouteName = "auto"
    report_mode: ReportMode = "concise"
    plan_mode: PlanMode = "auto"
    response_format: ResponseFormat = "text"
    include_sources: bool = True
    include_source_text: bool = False
    include_metadata: bool = True
    allow_degraded: bool = False
    max_sources: int = Field(default=5, ge=0, le=20)


class ItinerarySlot(BaseModel):
    start_time: str | None = None
    end_time: str | None = None
    title: str
    location: str | None = None
    activity: str | None = None
    transport_to_next: str | None = None
    estimated_cost: str | None = None
    ticket_info: str | None = None
    source_refs: list[str] = Field(default_factory=list)
    notes: str | None = None


class ItineraryDay(BaseModel):
    date_label: str
    slots: list[ItinerarySlot] = Field(default_factory=list)


class ItineraryPlan(BaseModel):
    days: list[ItineraryDay] = Field(default_factory=list)
    total_budget: str | None = None
    assumptions: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)


class QueryResponse(BaseModel):
    success: bool = True
    route: str
    answer: str
    sources: list[SourceItem] = Field(default_factory=list)
    metadata: QueryMetadata | dict[str, Any] = Field(default_factory=dict)
    structured_output: ItineraryPlan | dict[str, Any] | None = None
    error: ErrorDetail | None = None


class ErrorResponse(BaseModel):
    success: Literal[False] = False
    error: ErrorDetail


class DependencyCheck(BaseModel):
    status: Literal["ok", "failed", "skipped"]
    detail: str | None = None


class HealthResponse(BaseModel):
    status: Literal["ok"] = "ok"


class ReadyResponse(BaseModel):
    status: Literal["ok", "degraded"]
    checks: dict[str, DependencyCheck] = Field(default_factory=dict)


class GraphStatsResponse(BaseModel):
    documents: int
    chunks: int
    entities: int
    entity_relationships: int
    mentions: int
    relationships_with_refs: int
    communities: int
    summarized_communities: int

import time
from dataclasses import dataclass
from typing import Any
from uuid import uuid4

from fastapi import APIRouter, status
from fastapi.responses import JSONResponse

from agent_system.skills.registry import match_pre_query_skill
from server.deps import check_faiss_index, check_llm_config, check_neo4j, get_query_engine
from server.schemas import ErrorDetail, ErrorResponse, QueryMetadata, QueryRequest, QueryResponse, SourceItem
from server.trace_store import append_trace


router = APIRouter(tags=["query"])


GRAPH_ROUTES = {"auto", "local", "global", "hybrid", "agent"}
DEGRADABLE_ROUTES = {"local", "hybrid"}


@dataclass(frozen=True)
class RoutePreflight:
    route: str
    error: ErrorDetail | None = None
    degraded_from: str | None = None
    degradation_reason: str | None = None


def _to_source_item(source, include_text: bool) -> SourceItem:
    return SourceItem(
        doc_id=source.doc_id,
        chunk_id=source.chunk_id,
        source_path=source.source_path,
        file_name=source.file_name,
        title=getattr(source, "title", None),
        url=getattr(source, "url", None),
        published_at=getattr(source, "published_at", None),
        source_type=getattr(source, "source_type", None),
        chunk_index=source.chunk_index,
        page=source.page,
        section=source.section,
        text=source.text if include_text else None,
        score=source.score,
    )


def _to_response(
    result,
    request: QueryRequest,
    latency_ms: int,
    requested_route: str,
    trace_id: str,
    degraded_from: str | None = None,
    degradation_reason: str | None = None,
) -> QueryResponse:
    returned_sources = result.sources[: request.max_sources] if request.include_sources else []
    sources = [
        _to_source_item(source, include_text=request.include_source_text)
        for source in returned_sources
    ]

    metadata = QueryMetadata(
        trace_id=trace_id,
        latency_ms=latency_ms,
        requested_route=requested_route,
        actual_route=result.route,
        plan_mode=request.plan_mode,
        response_format=request.response_format,
        degraded=degraded_from is not None,
        degraded_from=degraded_from,
        degraded_to=result.route if degraded_from else None,
        degradation_reason=degradation_reason,
        source_count=len(result.sources),
        returned_source_count=len(sources),
        engine_metadata=result.metadata if request.include_metadata else {},
    )

    return QueryResponse(
        success=True,
        route=result.route,
        answer=result.answer,
        sources=sources,
        metadata=metadata if request.include_metadata else {"trace_id": trace_id},
        structured_output=getattr(result, "structured_output", None),
        error=None,
    )


def _append_query_trace(record: dict[str, Any]) -> None:
    try:
        append_trace(record)
    except Exception:
        # Trace persistence must not turn a query into a user-visible failure.
        pass


def _clarification_response(
    *,
    request: QueryRequest,
    requested_route: str,
    trace_id: str,
    latency_ms: int,
    clarification,
) -> QueryResponse:
    metadata = QueryMetadata(
        trace_id=trace_id,
        latency_ms=latency_ms,
        requested_route=requested_route,
        actual_route="clarification",
        plan_mode=request.plan_mode,
        response_format=request.response_format,
        source_count=0,
        returned_source_count=0,
        engine_metadata=clarification.metadata if request.include_metadata else {},
    )
    return QueryResponse(
        success=True,
        route="clarification",
        answer=clarification.answer,
        sources=[],
        metadata=metadata if request.include_metadata else {"trace_id": trace_id},
        error=None,
    )


def _error_response(
    *,
    code: str,
    message: str,
    detail: str | None,
    http_status: int,
    latency_ms: int,
    requested_route: str,
    trace_id: str,
    plan_mode: str = "auto",
    response_format: str = "text",
    actual_route: str | None = None,
    degraded_from: str | None = None,
    degraded_to: str | None = None,
    degradation_reason: str | None = None,
) -> JSONResponse:
    error = ErrorDetail(code=code, message=message, detail=detail)
    return JSONResponse(
        status_code=http_status,
        content={
            **ErrorResponse(error=error).model_dump(),
            "metadata": {
                "trace_id": trace_id,
                "latency_ms": latency_ms,
                "requested_route": requested_route,
                "actual_route": actual_route,
                "plan_mode": plan_mode,
                "response_format": response_format,
                "degraded": degraded_from is not None,
                "degraded_from": degraded_from,
                "degraded_to": degraded_to,
                "degradation_reason": degradation_reason,
            },
        },
    )


def _preflight_route(route: str, allow_degraded: bool) -> RoutePreflight:
    llm_ok, llm_detail = check_llm_config()
    if not llm_ok:
        return RoutePreflight(
            route=route,
            error=ErrorDetail(
                code="LLM_CONFIG_MISSING",
                message="LLM configuration is incomplete",
                detail=llm_detail,
            ),
        )

    if not check_faiss_index():
        return RoutePreflight(
            route=route,
            error=ErrorDetail(
                code="FAISS_INDEX_MISSING",
                message="FAISS index is missing",
                detail="Run `python manage.py build-index --source ./data` before querying.",
            ),
        )

    if route in GRAPH_ROUTES:
        try:
            check_neo4j()
        except Exception as exc:
            if allow_degraded and route in DEGRADABLE_ROUTES:
                return RoutePreflight(
                    route="vector",
                    degraded_from=route,
                    degradation_reason="NEO4J_UNAVAILABLE",
                )
            return RoutePreflight(
                route=route,
                error=ErrorDetail(
                    code="NEO4J_UNAVAILABLE",
                    message="Neo4j is unavailable for the requested route",
                    detail=str(exc),
                ),
            )

    return RoutePreflight(route=route)


def _query_with_route(request: QueryRequest, forced_route: str | None = None) -> QueryResponse:
    requested_route = forced_route or request.route
    trace_id = uuid4().hex
    started = time.perf_counter()

    clarification = match_pre_query_skill(
        request.question,
        route=requested_route,
        plan_mode=request.plan_mode,
    )
    if clarification is not None:
        latency_ms = int((time.perf_counter() - started) * 1000)
        _append_query_trace(
            {
                "trace_id": trace_id,
                "question": request.question,
                "requested_route": requested_route,
                "actual_route": "clarification",
                "latency_ms": latency_ms,
                "success": True,
                "error_code": None,
                "source_count": 0,
                "degraded": False,
                "degraded_from": None,
                "degraded_to": None,
                "degradation_reason": None,
                "skill": clarification.metadata.get("skill"),
                "skill_spec": clarification.metadata.get("skill_spec"),
                "clarification_required": True,
                "clarification_type": clarification.metadata.get("clarification_type"),
                "plan_mode": request.plan_mode,
                "response_format": request.response_format,
            }
        )
        return _clarification_response(
            request=request,
            requested_route=requested_route,
            trace_id=trace_id,
            latency_ms=latency_ms,
            clarification=clarification,
        )

    preflight = _preflight_route(requested_route, request.allow_degraded)
    if preflight.error:
        latency_ms = int((time.perf_counter() - started) * 1000)
        _append_query_trace(
            {
                "trace_id": trace_id,
                "question": request.question,
                "requested_route": requested_route,
                "actual_route": None,
                "latency_ms": latency_ms,
                "success": False,
                "error_code": preflight.error.code,
                "source_count": 0,
                "degraded": False,
                "degraded_from": None,
                "degraded_to": None,
                "degradation_reason": None,
                "plan_mode": request.plan_mode,
                "response_format": request.response_format,
            }
        )
        return _error_response(
            code=preflight.error.code,
            message=preflight.error.message,
            detail=preflight.error.detail,
            http_status=status.HTTP_503_SERVICE_UNAVAILABLE,
            latency_ms=latency_ms,
            requested_route=requested_route,
            trace_id=trace_id,
            plan_mode=request.plan_mode,
            response_format=request.response_format,
        )

    route = preflight.route
    try:
        result = get_query_engine().ask(
            request.question,
            route=route,
            report_mode=request.report_mode,
            plan_mode=request.plan_mode,
            response_format=request.response_format,
        )
        latency_ms = int((time.perf_counter() - started) * 1000)
        _append_query_trace(
            {
                "trace_id": trace_id,
                "question": request.question,
                "requested_route": requested_route,
                "actual_route": result.route,
                "latency_ms": latency_ms,
                "success": True,
                "error_code": None,
                "source_count": len(result.sources),
                "degraded": preflight.degraded_from is not None,
                "degraded_from": preflight.degraded_from,
                "degraded_to": result.route if preflight.degraded_from else None,
                "degradation_reason": preflight.degradation_reason,
                "plan_mode": request.plan_mode,
                "response_format": request.response_format,
                "agent_trace": result.metadata.get("agent_trace") if isinstance(result.metadata, dict) else None,
                "structured_output": getattr(result, "structured_output", None),
                "itinerary_validation": (
                    result.metadata.get("itinerary_validation")
                    if isinstance(result.metadata, dict)
                    else None
                ),
            }
        )
        return _to_response(
            result,
            request,
            latency_ms,
            requested_route=requested_route,
            trace_id=trace_id,
            degraded_from=preflight.degraded_from,
            degradation_reason=preflight.degradation_reason,
        )
    except Exception as exc:
        latency_ms = int((time.perf_counter() - started) * 1000)
        error_code = "AGENT_EXECUTION_FAILED" if requested_route == "agent" else "QUERY_FAILED"
        _append_query_trace(
            {
                "trace_id": trace_id,
                "question": request.question,
                "requested_route": requested_route,
                "actual_route": route,
                "latency_ms": latency_ms,
                "success": False,
                "error_code": error_code,
                "source_count": 0,
                "degraded": preflight.degraded_from is not None,
                "degraded_from": preflight.degraded_from,
                "degraded_to": route if preflight.degraded_from else None,
                "degradation_reason": preflight.degradation_reason,
                "plan_mode": request.plan_mode,
                "response_format": request.response_format,
            }
        )
        return _error_response(
            code=error_code,
            message="Agent execution failed" if requested_route == "agent" else "Query execution failed",
            detail=str(exc),
            http_status=status.HTTP_500_INTERNAL_SERVER_ERROR,
            latency_ms=latency_ms,
            requested_route=requested_route,
            trace_id=trace_id,
            plan_mode=request.plan_mode,
            response_format=request.response_format,
            actual_route=route,
            degraded_from=preflight.degraded_from,
            degraded_to=route if preflight.degraded_from else None,
            degradation_reason=preflight.degradation_reason,
        )


ERROR_RESPONSES = {
    500: {"model": ErrorResponse},
    503: {"model": ErrorResponse},
}


@router.post("/query", response_model=QueryResponse, responses=ERROR_RESPONSES)
def query(request: QueryRequest):
    return _query_with_route(request)


@router.post("/agent/query", response_model=QueryResponse, responses=ERROR_RESPONSES)
def agent_query(request: QueryRequest):
    return _query_with_route(request, forced_route="agent")

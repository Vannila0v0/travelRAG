from fastapi import APIRouter, HTTPException, status

from server.deps import get_query_engine
from server.schemas import QueryRequest, QueryResponse, SourceModel


router = APIRouter(tags=["query"])


def _to_response(result, request: QueryRequest) -> QueryResponse:
    sources = []
    for source in result.sources[: request.max_sources]:
        sources.append(
            SourceModel(
                doc_id=source.doc_id,
                chunk_id=source.chunk_id,
                source_path=source.source_path,
                file_name=source.file_name,
                chunk_index=source.chunk_index,
                page=source.page,
                section=source.section,
                text=source.text if request.include_source_text else None,
                score=source.score,
            )
        )

    return QueryResponse(
        route=result.route,
        answer=result.answer,
        sources=sources,
        metadata=result.metadata,
    )


@router.post("/query", response_model=QueryResponse)
def query(request: QueryRequest):
    try:
        result = get_query_engine().ask(request.question, route=request.route)
        return _to_response(result, request)
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(exc),
        ) from exc


@router.post("/agent/query", response_model=QueryResponse)
def agent_query(request: QueryRequest):
    try:
        result = get_query_engine().ask(request.question, route="agent")
        return _to_response(result, request)
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(exc),
        ) from exc

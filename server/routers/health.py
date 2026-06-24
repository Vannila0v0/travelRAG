from fastapi import APIRouter

from server.deps import (
    check_embedding_config,
    check_faiss_index,
    check_llm_config,
    check_neo4j,
)
from server.schemas import DependencyCheck, HealthResponse, ReadyResponse


router = APIRouter(tags=["health"])


@router.get("/health", response_model=HealthResponse)
def health():
    return HealthResponse()


@router.get("/ready", response_model=ReadyResponse)
def ready():
    checks: dict[str, DependencyCheck] = {}

    try:
        check_neo4j()
        checks["neo4j"] = DependencyCheck(status="ok")
    except Exception as exc:
        checks["neo4j"] = DependencyCheck(status="failed", detail=str(exc))

    if check_faiss_index():
        checks["faiss_index"] = DependencyCheck(status="ok")
    else:
        checks["faiss_index"] = DependencyCheck(
            status="failed",
            detail="FAISS index files not found",
        )

    llm_ok, llm_detail = check_llm_config()
    checks["llm_config"] = DependencyCheck(
        status="ok" if llm_ok else "failed",
        detail=llm_detail,
    )

    embedding_ok, embedding_detail = check_embedding_config()
    checks["embedding_config"] = DependencyCheck(
        status="ok" if embedding_ok else "failed",
        detail=embedding_detail,
    )

    status = "ok" if all(check.status == "ok" for check in checks.values()) else "degraded"
    return ReadyResponse(status=status, checks=checks)

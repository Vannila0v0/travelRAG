from fastapi import APIRouter

from server.deps import check_faiss_index, check_neo4j
from server.schemas import HealthResponse


router = APIRouter(tags=["health"])


@router.get("/health", response_model=HealthResponse)
def health():
    details = {}
    neo4j_ok = False
    try:
        neo4j_ok = check_neo4j()
    except Exception as exc:
        details["neo4j_error"] = str(exc)

    faiss_ok = check_faiss_index()
    if not faiss_ok:
        details["faiss_error"] = "FAISS index files not found"

    return HealthResponse(
        status="ok" if neo4j_ok and faiss_ok else "degraded",
        neo4j=neo4j_ok,
        faiss_index=faiss_ok,
        details=details,
    )

from fastapi import APIRouter

from server.deps import graph_stats
from server.schemas import GraphStatsResponse


router = APIRouter(prefix="/graph", tags=["graph"])


@router.get("/stats", response_model=GraphStatsResponse)
def stats():
    return GraphStatsResponse(**graph_stats())

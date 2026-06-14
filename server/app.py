from fastapi import FastAPI

from server.deps import close_query_engine
from server.routers import graph, health, query


app = FastAPI(
    title="Local RAG / GraphRAG Tourism API",
    version="0.1.0",
)


app.include_router(health.router)
app.include_router(graph.router)
app.include_router(query.router)


@app.on_event("shutdown")
def shutdown_event():
    close_query_engine()

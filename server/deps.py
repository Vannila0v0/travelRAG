import os
from pathlib import Path
from threading import Lock

from core.config import neo4j_password, neo4j_url, neo4j_user
from core.neo4j_handler import Neo4jHandler
from query_engine import QueryEngine


DEFAULT_INDEX_DIR = Path(".cache/faiss_index")

_engine = None
_engine_lock = Lock()


def get_query_engine() -> QueryEngine:
    global _engine
    with _engine_lock:
        if _engine is None:
            _engine = QueryEngine(index_dir=str(DEFAULT_INDEX_DIR))
        return _engine


def close_query_engine() -> None:
    global _engine
    with _engine_lock:
        if _engine is not None:
            _engine.close()
            _engine = None


def check_neo4j() -> bool:
    handler = Neo4jHandler(neo4j_url, neo4j_user, neo4j_password)
    try:
        with handler.driver.session() as session:
            session.run("RETURN 1").single()
        return True
    finally:
        handler.close()


def check_faiss_index(index_dir: Path = DEFAULT_INDEX_DIR) -> bool:
    return (index_dir / "index.faiss").exists() and (index_dir / "index.pkl").exists()


def check_llm_config() -> tuple[bool, str | None]:
    api_key = os.getenv("DEEPSEEK_API_KEY", "").strip()
    if not api_key or api_key == "your_deepseek_api_key":
        return False, "DEEPSEEK_API_KEY is missing or still uses the placeholder value"

    base_url = os.getenv("DEEPSEEK_BASE_URL", "").strip()
    if not base_url:
        return False, "DEEPSEEK_BASE_URL is missing"

    model = os.getenv("DEEPSEEK_MODEL", "").strip()
    if not model:
        return False, "DEEPSEEK_MODEL is missing"

    return True, None


def check_embedding_config() -> tuple[bool, str | None]:
    model_path = os.getenv("EMBEDDING_MODEL_PATH", "").strip()
    if not model_path:
        return False, "EMBEDDING_MODEL_PATH is missing"

    device = os.getenv("MODEL_DEVICE", "cpu").strip()
    if not device:
        return False, "MODEL_DEVICE is missing"

    return True, None


def graph_stats() -> dict[str, int]:
    handler = Neo4jHandler(neo4j_url, neo4j_user, neo4j_password)
    queries = {
        "documents": "MATCH (d:Document) RETURN count(d) AS c",
        "chunks": "MATCH (c:Chunk) RETURN count(c) AS c",
        "entities": "MATCH (e:Entity) RETURN count(e) AS c",
        "entity_relationships": "MATCH (:Entity)-[r]->(:Entity) RETURN count(r) AS c",
        "mentions": "MATCH (:Entity)-[:MENTIONED_IN]->(:Chunk) RETURN count(*) AS c",
        "relationships_with_refs": "MATCH (:Entity)-[r]->(:Entity) WHERE r.source_ref_ids IS NOT NULL RETURN count(r) AS c",
        "communities": "MATCH (c:Community) RETURN count(c) AS c",
        "summarized_communities": "MATCH (c:Community) WHERE c.summary IS NOT NULL RETURN count(c) AS c",
    }
    try:
        with handler.driver.session() as session:
            return {
                name: session.run(query).single()["c"]
                for name, query in queries.items()
            }
    finally:
        handler.close()

from dataclasses import asdict

from agent_system.integration.llm_factory import get_text_llm
from core.config import neo4j_password, neo4j_url, neo4j_user
from core.neo4j_handler import Neo4jHandler
from .graph_global import GraphGlobalQueryEngine
from .graph_local import GraphLocalQueryEngine
from .hybrid_query import HybridQueryEngine
from .source_utils import dedupe_sources
from .types import QueryResult, Source
from .vector_query import VectorQueryEngine


HYBRID_KEYWORDS = (
    "多少钱", "票价", "门票", "价格", "优惠", "政策", "规则", "退票", "改签",
    "免票", "半价", "儿童票", "儿童", "船票", "乘坐", "限制", "开放时间",
    "怎么去", "交通", "码头", "电话",
)
GLOBAL_PLANNING_KEYWORDS = (
    "规划", "路线", "行程", "一日游", "两天", "多景点", "怎么玩", "怎么游",
    "怎么安排", "安排",
)
GLOBAL_SUMMARY_KEYWORDS = ("整体", "概况", "比较", "推荐", "总结", "适合")
BROAD_AREA_KEYWORDS = ("桂林", "市区", "桂林市区")
SPECIFIC_DETAIL_PATTERNS = ("特色景点", "设施", "景观")
SPECIFIC_ENTITY_HINTS = (
    "两江四湖", "银子岩", "银子岩溶洞", "遇龙河", "漓江精华游船", "漓江游船",
    "靖江王府", "象鼻山",
)


class QueryEngine:
    def __init__(self, index_dir: str = ".cache/faiss_index", llm=None):
        self.llm = llm or get_text_llm()
        self.neo4j = Neo4jHandler(neo4j_url, neo4j_user, neo4j_password)
        self.vector = VectorQueryEngine(index_dir=index_dir, llm=self.llm)
        self.local = GraphLocalQueryEngine(llm=self.llm, neo4j_handler=self.neo4j)
        self.global_search = GraphGlobalQueryEngine(llm=self.llm, neo4j_handler=self.neo4j)
        self.hybrid = HybridQueryEngine(self.vector, self.local, llm=self.llm)

    def close(self):
        self.neo4j.close()

    def route(self, query: str, forced_route: str | None = None) -> str:
        if forced_route and forced_route != "auto":
            return forced_route

        if any(keyword in query for keyword in HYBRID_KEYWORDS):
            return "hybrid"

        if any(keyword in query for keyword in GLOBAL_PLANNING_KEYWORDS):
            return "global"

        if (
            any(keyword in query for keyword in BROAD_AREA_KEYWORDS)
            and any(keyword in query for keyword in GLOBAL_SUMMARY_KEYWORDS + ("有哪些",))
        ):
            return "global"

        if (
            any(entity in query for entity in SPECIFIC_ENTITY_HINTS)
            and any(pattern in query for pattern in SPECIFIC_DETAIL_PATTERNS)
        ):
            return "hybrid"

        if any(keyword in query for keyword in GLOBAL_SUMMARY_KEYWORDS):
            return "global"

        return "vector"

    def ask(self, query: str, route: str = "auto") -> QueryResult:
        selected = self.route(query, route)
        if selected == "vector":
            return self.vector.search(query)
        if selected == "local":
            return self.local.search(query)
        if selected == "global":
            return self.global_search.search(query)
        if selected == "hybrid":
            return self.hybrid.search(query)
        if selected == "agent":
            from agent_system.orchestrator import MultiAgentOrchestrator

            orchestrator = MultiAgentOrchestrator()
            state = orchestrator.run(query)
            tasks = []
            if state.plan and state.plan.task_graph:
                tasks = [
                    asdict(task)
                    if hasattr(task, "__dataclass_fields__")
                    else (task.model_dump() if hasattr(task, "model_dump") else task.dict())
                    for task in state.plan.task_graph.nodes
                ]
            cache_hits = sum(
                1 for record in state.execution_records
                if record.tool_metadata.get("cache_hit") is True
            )
            cache_misses = sum(
                1 for record in state.execution_records
                if record.tool_metadata.get("cache_hit") is False
            )
            sources = [
                Source(
                    doc_id=source.get("doc_id"),
                    chunk_id=source.get("chunk_id"),
                    source_path=source.get("source_path"),
                    file_name=source.get("file_name"),
                    chunk_index=source.get("chunk_index"),
                    page=source.get("page"),
                    section=source.get("section"),
                    text=source.get("text"),
                    score=source.get("score"),
                )
                for source in getattr(state, "sources", [])
                if isinstance(source, dict)
            ]
            return QueryResult(
                answer=state.final_report or "未能生成多智能体报告。",
                route="agent",
                sources=dedupe_sources(sources),
                metadata={
                    "tasks": tasks,
                    "tool_cache": {
                        "hits": cache_hits,
                        "misses": cache_misses,
                        "total": cache_hits + cache_misses,
                    },
                },
            )
        raise ValueError(f"Unknown query route: {selected}")

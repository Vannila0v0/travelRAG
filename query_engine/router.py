from dataclasses import asdict
from typing import Any

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


def _dump_model(value: Any) -> dict[str, Any]:
    if hasattr(value, "__dataclass_fields__"):
        return asdict(value)
    if hasattr(value, "model_dump"):
        return value.model_dump()
    if hasattr(value, "dict"):
        return value.dict()
    return dict(value)


def _build_agent_metadata(state) -> dict[str, Any]:
    tasks = []
    if state.plan and state.plan.task_graph:
        tasks = [_dump_model(task) for task in state.plan.task_graph.nodes]

    records_by_task_id = {
        record.task_id: record
        for record in state.execution_records
    }

    traced_tasks = []
    for task in tasks:
        task_id = task.get("task_id")
        record = records_by_task_id.get(task_id)
        latency_ms = None
        tool_name = None
        source_count = 0
        record_route = None
        cache_hit = None
        error = None

        if record is not None:
            latency_ms = int(record.metadata.latency_seconds * 1000)
            tool_name = record.inputs.get("task_type") or task.get("task_type")
            source_count = len(record.sources)
            record_route = record.route
            cache_hit = record.tool_metadata.get("cache_hit")
            error = record.metadata.error

        traced_tasks.append(
            {
                "task_id": task_id,
                "task_type": task.get("task_type"),
                "status": task.get("status"),
                "latency_ms": latency_ms,
                "tool_name": tool_name,
                "source_count": source_count,
                "route": record_route,
                "cache_hit": cache_hit,
                "error": error,
            }
        )

    cache_hits = sum(
        1 for record in state.execution_records
        if record.tool_metadata.get("cache_hit") is True
    )
    cache_misses = sum(
        1 for record in state.execution_records
        if record.tool_metadata.get("cache_hit") is False
    )

    agent_trace = {
        **getattr(state, "agent_trace", {}),
        "task_count": len(tasks),
        "tasks": traced_tasks,
        "cache_hits": cache_hits,
        "cache_misses": cache_misses,
    }

    return {
        "tasks": tasks,
        "plan_mode": getattr(state, "plan_mode", "auto"),
        "agent_trace": agent_trace,
        "tool_cache": {
            "hits": cache_hits,
            "misses": cache_misses,
            "total": cache_hits + cache_misses,
        },
    }


class QueryEngine:
    def __init__(self, index_dir: str = ".cache/faiss_index", llm=None):
        self.llm = llm or get_text_llm()
        self.index_dir = index_dir
        self._neo4j = None
        self._vector = None
        self._local = None
        self._global_search = None
        self._hybrid = None

    def close(self):
        if self._neo4j is not None:
            self._neo4j.close()
            self._neo4j = None

    @property
    def neo4j(self):
        if self._neo4j is None:
            self._neo4j = Neo4jHandler(neo4j_url, neo4j_user, neo4j_password)
        return self._neo4j

    @property
    def vector(self):
        if self._vector is None:
            self._vector = VectorQueryEngine(index_dir=self.index_dir, llm=self.llm)
        return self._vector

    @property
    def local(self):
        if self._local is None:
            self._local = GraphLocalQueryEngine(llm=self.llm, neo4j_handler=self.neo4j)
        return self._local

    @property
    def global_search(self):
        if self._global_search is None:
            self._global_search = GraphGlobalQueryEngine(llm=self.llm, neo4j_handler=self.neo4j)
        return self._global_search

    @property
    def hybrid(self):
        if self._hybrid is None:
            self._hybrid = HybridQueryEngine(self.vector, self.local, llm=self.llm)
        return self._hybrid

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

    def ask(
        self,
        query: str,
        route: str = "auto",
        report_mode: str = "concise",
        plan_mode: str = "auto",
        response_format: str = "text",
    ) -> QueryResult:
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
            state = orchestrator.run(query, report_mode=report_mode, plan_mode=plan_mode)
            agent_metadata = _build_agent_metadata(state)
            structured_output = None
            if response_format == "itinerary" and plan_mode == "detailed_itinerary":
                from agent_system.reporter.itinerary_builder import ItineraryBuilder
                from agent_system.reporter.itinerary_validator import ItineraryValidator

                itinerary_builder = ItineraryBuilder()
                structured_output = itinerary_builder.build(state)
                validation = ItineraryValidator().validate(structured_output)
                agent_metadata["structured_output_type"] = "itinerary"
                agent_metadata["itinerary_metrics"] = itinerary_builder.last_metrics
                agent_metadata["itinerary_validation"] = validation
            elif response_format == "itinerary":
                agent_metadata["structured_output_type"] = None
                agent_metadata["structured_output_skipped_reason"] = (
                    "itinerary requires route=agent and plan_mode=detailed_itinerary"
                )
            sources = [
                Source(
                    doc_id=source.get("doc_id"),
                    chunk_id=source.get("chunk_id"),
                    source_path=source.get("source_path"),
                    file_name=source.get("file_name"),
                    title=source.get("title"),
                    url=source.get("url"),
                    published_at=source.get("published_at"),
                    source_type=source.get("source_type"),
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
                metadata=agent_metadata,
                structured_output=structured_output,
            )
        raise ValueError(f"Unknown query route: {selected}")

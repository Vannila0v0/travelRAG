import copy
import logging
import os
from threading import Lock
from typing import Any

from agent_system.integration.llm_factory import DEEPSEEK_MODEL, LLM_TEMPERATURE
from graphrag_agent.cache_manager.config import CacheConfig
from graphrag_agent.cache_manager.manager import CacheManager
from query_engine import QueryEngine


TOOL_CACHE_SCHEMA_VERSION = "tool-cache-v1"
TOOL_CACHE_PROMPT_VERSION = "tool-prompt-v1"
_LOGGER = logging.getLogger(__name__)

_QUERY_ENGINE: QueryEngine | None = None
_QUERY_ENGINE_LOCK = Lock()

_TOOL_CACHE: CacheManager | None = None
_TOOL_CACHE_LOCK = Lock()


def _agent_tool_cache_enabled() -> bool:
    return os.getenv("AGENT_TOOL_CACHE_ENABLED", "true").lower() == "true"


def get_query_engine() -> QueryEngine:
    global _QUERY_ENGINE
    if _QUERY_ENGINE is None:
        with _QUERY_ENGINE_LOCK:
            if _QUERY_ENGINE is None:
                _QUERY_ENGINE = QueryEngine()
    return _QUERY_ENGINE


def get_tool_cache() -> CacheManager:
    global _TOOL_CACHE
    if _TOOL_CACHE is None:
        with _TOOL_CACHE_LOCK:
            if _TOOL_CACHE is None:
                _TOOL_CACHE = CacheManager(
                    CacheConfig(
                        backend_type=os.getenv("AGENT_TOOL_CACHE_BACKEND", "hybrid"),
                        base_dir=os.getenv("AGENT_TOOL_CACHE_DIR", ".cache/tool_cache"),
                        default_ttl=int(os.getenv("AGENT_TOOL_CACHE_TTL", os.getenv("CACHE_DEFAULT_TTL", "86400"))),
                        max_memory_items=int(os.getenv("AGENT_TOOL_CACHE_MAX_ITEMS", os.getenv("CACHE_MAX_MEMORY_ITEMS", "1000"))),
                        enable_vector_match=False,
                    )
                )
    return _TOOL_CACHE


def _normalize_entities(entities: Any) -> list[str]:
    if not entities:
        return []
    normalized = set()
    keep_raw_keywords = (
        "两江四湖", "日月双塔", "象鼻山", "靖江王府", "独秀峰",
        "银子岩", "遇龙河", "漓江", "龙胜温泉",
    )
    if isinstance(entities, str):
        entities = [entities]
    try:
        for item in entities:
            text = str(item).strip()
            if not text:
                continue
            keep_raw = any(keyword in text for keyword in keep_raw_keywords)
            if "桂林市区" in text:
                normalized.add("桂林市区")
            if "景点" in text:
                normalized.add("景点")
            if "交通" in text or "公交" in text:
                normalized.add("交通")
            if "美食" in text or "米粉" in text or "餐" in text:
                normalized.add("餐饮")
            if keep_raw:
                normalized.add(text)
        if "靖江王府" in normalized and "独秀峰" in normalized:
            normalized.remove("独秀峰")
        return sorted(normalized)
    except TypeError:
        return [str(entities)]


def _intent_tags(text: str) -> list[str]:
    tag_rules = {
        "overview": ("核心", "经典", "景点", "概览", "顺序", "列表", "特色"),
        "ticket": ("票价", "门票", "价格", "费用", "预算", "优惠"),
        "time": ("开放时间", "运营时间", "游玩时长", "时间"),
        "traffic": ("交通", "公交", "打车", "步行", "码头", "车站", "桂林站", "桂林北站"),
        "food": ("午餐", "晚餐", "餐", "美食", "米粉", "啤酒鱼", "消费", "正阳步行街", "东西巷"),
        "route_check": ("校验", "合理", "衔接", "顺路", "不走回头", "综合"),
    }
    tags = [
        tag
        for tag, keywords in tag_rules.items()
        if any(keyword in text for keyword in keywords)
    ]
    return tags or ["general"]


def _mark_cache_hit(result: dict[str, Any], is_hit: bool) -> dict[str, Any]:
    payload = copy.deepcopy(result)
    data = payload.setdefault("data", {})
    metadata = data.setdefault("metadata", {})
    metadata["cache_hit"] = is_hit
    metadata["cache_layer"] = "agent_tool"
    metadata["cache_schema_version"] = TOOL_CACHE_SCHEMA_VERSION
    return payload


class ToolAdapter:
    def __init__(self, route: str, task_type: str):
        self.route = route
        self.task_type = task_type
        self.name = task_type

    def _cache_context(self, payload: dict[str, Any]) -> dict[str, Any]:
        description = str(payload.get("query") or payload.get("description") or "")
        return {
            "cache_schema_version": TOOL_CACHE_SCHEMA_VERSION,
            "prompt_version": os.getenv("AGENT_TOOL_PROMPT_VERSION", TOOL_CACHE_PROMPT_VERSION),
            "data_version": os.getenv("KNOWLEDGE_BASE_VERSION", "default"),
            "route": self.route,
            "task_type": payload.get("task_type") or self.task_type,
            "agent_role": f"tool:{payload.get('task_type') or self.task_type}",
            "entities": _normalize_entities(payload.get("entities")),
            "intent_tags": _intent_tags(description),
            "model": DEEPSEEK_MODEL,
            "temperature": LLM_TEMPERATURE,
        }

    def _cache_query(self, query: str, cache_context: dict[str, Any]) -> str:
        entities = ",".join(cache_context["entities"])
        intent_tags = ",".join(cache_context["intent_tags"])
        return (
            f"{TOOL_CACHE_SCHEMA_VERSION}|"
            f"route={self.route}|"
            f"task_type={cache_context['task_type']}|"
            f"entities={entities}|"
            f"intent={intent_tags}"
        )

    def structured_search(self, payload: dict[str, Any]) -> dict[str, Any]:
        query = payload.get("query") or payload.get("description")
        if not query and payload.get("entities"):
            query = ",".join(_normalize_entities(payload["entities"]))

        if not query:
            return {"success": False, "error": "empty query"}

        cache_context = self._cache_context(payload)
        cache_query = self._cache_query(query, cache_context)
        if _agent_tool_cache_enabled():
            cached_result = get_tool_cache().get(cache_query, **cache_context)
            if isinstance(cached_result, dict):
                _LOGGER.info("Agent tool cache HIT: %s", cache_query)
                return _mark_cache_hit(cached_result, True)

        result = get_query_engine().ask(query, route=self.route)
        tool_result = {
            "success": True,
            "answer": result.answer,
            "data": {
                "route": result.route,
                "sources": [
                    {
                        "doc_id": source.doc_id,
                        "chunk_id": source.chunk_id,
                        "source_path": source.source_path,
                        "file_name": source.file_name,
                        "chunk_index": source.chunk_index,
                        "page": source.page,
                        "section": source.section,
                        "text": source.text,
                        "score": source.score,
                    }
                    for source in result.sources
                ],
                "metadata": {
                    **result.metadata,
                    "cache_hit": False,
                    "cache_layer": "agent_tool",
                    "cache_schema_version": TOOL_CACHE_SCHEMA_VERSION,
                },
            },
        }

        if _agent_tool_cache_enabled():
            _LOGGER.info("Agent tool cache SET: %s", cache_query)
            get_tool_cache().set(cache_query, tool_result, **cache_context)

        return tool_result


TOOL_REGISTRY = {
    "local_search": lambda: ToolAdapter("hybrid", "local_search"),
    "global_search": lambda: ToolAdapter("global", "global_search"),
    "reflection": lambda: ToolAdapter("hybrid", "reflection"),
}

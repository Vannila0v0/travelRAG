import copy
import itertools
import ipaddress
import json
import logging
import math
import os
import re
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta, timezone
from html import unescape
from html.parser import HTMLParser
from threading import Lock
from typing import Any
from urllib.parse import urljoin, urlparse

import httpx

from agent_system.integration.llm_factory import DEEPSEEK_MODEL, LLM_TEMPERATURE, get_llm_model
from graphrag_agent.cache_manager.config import CacheConfig
from graphrag_agent.cache_manager.manager import CacheManager
from query_engine import QueryEngine


TOOL_CACHE_SCHEMA_VERSION = "tool-cache-v1"
TOOL_CACHE_PROMPT_VERSION = "tool-prompt-v1"
_LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class PlanPolicy:
    importance: str = "normal"
    merge_strategy: str = "none"
    max_instances: int | None = None
    dedupe_keys: tuple[str, ...] = ()
    group: str | None = None
    requires_previous: tuple[str, ...] = ()
    requires_followup: tuple[str, ...] = ()
    drop_priority: int = 50
    realtime_sensitive: bool = False


@dataclass(frozen=True)
class ToolSpec:
    name: str
    description: str
    use_when: list[str] = field(default_factory=list)
    avoid_when: list[str] = field(default_factory=list)
    parameters: dict[str, str] = field(default_factory=dict)
    plan_policy: PlanPolicy = field(default_factory=PlanPolicy)


def format_tool_specs_for_planner() -> str:
    lines = []
    for index, spec in enumerate(TOOL_SPECS.values(), start=1):
        lines.append(f"{index}. **{spec.name}**: {spec.description}")
        if spec.use_when:
            lines.append(f"   - 适用场景: {'；'.join(spec.use_when)}")
        if spec.avoid_when:
            lines.append(f"   - 避免场景: {'；'.join(spec.avoid_when)}")
        if spec.parameters:
            params = "；".join(f"{name}: {desc}" for name, desc in spec.parameters.items())
            lines.append(f"   - 参数: {params}")
    return "\n".join(lines)


REFLECTION_PROMPT = """你是一个多智能体任务校验与综合助手。请只基于已有子任务结果做反思，不要引入新资料。

**当前反思任务**:
{description}

**已有子任务结果**:
{evidence}

请输出简洁的 Markdown，完成以下内容：
1. 综合已有结果，给出可执行结论。
2. 检查是否存在时间、票价、交通、路线或信息冲突。
3. 标出仍缺失或需要用户二次确认的信息。
4. 不要编造已有证据中没有的信息。
"""

SOURCE_SELECT_PROMPT = """你是一个来源选择助手。请只从候选网页来源中选择最适合回答用户问题的一条。

**用户问题**:
{query}

**当前选择任务**:
{description}

**候选来源**:
{sources_json}

请严格输出 JSON，不要输出 Markdown，不要添加解释文字：
{{
  "selected_source_index": 1,
  "confidence": 0.0,
  "reason": "选择理由"
}}

要求：
1. selected_source_index 必须是候选来源中的 source_index。
2. 优先选择官方、权威、标题和摘要与用户问题最匹配的来源。
3. 如果候选都不理想，也必须选择相对最合适的一条，并在 reason 里说明风险。
"""

WEB_SEARCH_UNCONFIGURED_MESSAGE = """Web Search 工具尚未配置。请设置 WEB_SEARCH_PROVIDER 后重试。

当前任务需要实时网页信息，不能只依赖本地知识库。"""

_MCP_URL = "https://mcp.exa.ai/mcp"

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


def _dedupe_source_dicts(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    seen = set()
    sources = []
    for record in records:
        for source in record.get("sources", []) or []:
            if not isinstance(source, dict):
                continue
            key = (
                source.get("chunk_id")
                or (source.get("doc_id"), source.get("source_path"), source.get("file_name"), source.get("chunk_index"))
                or tuple(sorted((k, str(v)) for k, v in source.items() if v is not None))
            )
            if key in seen:
                continue
            seen.add(key)
            sources.append(source)
    return sources


class ReflectionTool:
    def __init__(self, llm=None):
        self.name = "reflection"
        self._llm = llm or get_llm_model()

    def structured_search(self, payload: dict[str, Any]) -> dict[str, Any]:
        previous_records = [
            record for record in payload.get("previous_records", [])
            if isinstance(record, dict)
        ]
        if not previous_records:
            return {
                "success": True,
                "answer": "目前没有可用于反思的前序任务结果。",
                "data": {
                    "route": "reflection",
                    "sources": [],
                    "metadata": {
                        "cache_hit": False,
                        "cache_layer": "agent_reflection",
                        "records_used": 0,
                        "llm_calls": 0,
                    },
                },
            }

        evidence = self._format_previous_records(previous_records)
        prompt = REFLECTION_PROMPT.format(
            description=payload.get("description") or "综合已有任务结果",
            evidence=evidence[:6000],
        )
        response = self._llm.invoke(prompt)
        answer = str(response.content if hasattr(response, "content") else response)
        sources = _dedupe_source_dicts(previous_records)
        return {
            "success": True,
            "answer": answer,
            "data": {
                "route": "reflection",
                "sources": sources,
                "metadata": {
                    "cache_hit": False,
                    "cache_layer": "agent_reflection",
                    "records_used": len(previous_records),
                    "llm_calls": 1,
                },
            },
        }

    @staticmethod
    def _format_previous_records(records: list[dict[str, Any]]) -> str:
        chunks = []
        for index, record in enumerate(records, start=1):
            inputs = record.get("inputs") or {}
            metadata = record.get("metadata") or {}
            output = str(record.get("output") or "").strip()
            chunks.append(
                "\n".join(
                    [
                        f"[record_{index}] task_id={record.get('task_id')} route={record.get('route')}",
                        f"task_type={inputs.get('task_type')} description={inputs.get('description')}",
                        f"error={metadata.get('error')}",
                        f"output={output[:1200]}",
                    ]
                )
            )
        return "\n\n".join(chunks)


class SourceSelectTool:
    def __init__(self, llm=None):
        self.name = "source_select"
        self._llm = llm or get_llm_model()
        self.max_candidates = int(os.getenv("SOURCE_SELECT_MAX_CANDIDATES", "8"))

    def structured_search(self, payload: dict[str, Any]) -> dict[str, Any]:
        previous_records = [
            record for record in payload.get("previous_records", [])
            if isinstance(record, dict)
        ]
        candidates = self._collect_url_candidates(previous_records)
        if not candidates:
            return {
                "success": False,
                "answer": json.dumps(
                    {
                        "error": "missing_url_sources",
                        "message": "source_select 需要前序任务提供带 URL 的 sources。",
                    },
                    ensure_ascii=False,
                    indent=2,
                ),
                "error": "source_select 需要前序任务提供带 URL 的 sources。",
                "data": {
                    "route": "source_select",
                    "sources": [],
                    "metadata": {
                        "candidate_count": 0,
                        "selected_source_index": None,
                        "confidence": 0.0,
                    },
                },
            }

        query = str(payload.get("query") or payload.get("original_query") or payload.get("description") or "").strip()
        prompt = SOURCE_SELECT_PROMPT.format(
            query=query,
            description=payload.get("description") or "选择最合适的网页来源",
            sources_json=json.dumps(candidates[: self.max_candidates], ensure_ascii=False, indent=2),
        )
        response = self._llm.invoke(prompt)
        raw_text = str(response.content if hasattr(response, "content") else response).strip()
        try:
            parsed = self._parse_json(raw_text)
            selected_source_index = int(parsed.get("selected_source_index"))
        except Exception as exc:
            return {
                "success": False,
                "answer": raw_text,
                "error": f"source_select JSON parse failed: {exc}",
                "data": {
                    "route": "source_select",
                    "sources": [],
                    "metadata": {
                        "candidate_count": len(candidates),
                        "selected_source_index": None,
                        "confidence": 0.0,
                        "raw_output": raw_text[:1000],
                    },
                },
            }

        candidate_by_index = {item["source_index"]: item for item in candidates}
        selected = candidate_by_index.get(selected_source_index)
        if not selected:
            return {
                "success": False,
                "answer": raw_text,
                "error": f"selected_source_index={selected_source_index} is not available",
                "data": {
                    "route": "source_select",
                    "sources": [],
                    "metadata": {
                        "candidate_count": len(candidates),
                        "selected_source_index": selected_source_index,
                        "confidence": self._coerce_confidence(parsed.get("confidence")),
                        "reason": parsed.get("reason"),
                    },
                },
            }

        answer_payload = {
            "selected_source_index": selected_source_index,
            "selected_url": selected.get("url"),
            "selected_title": selected.get("title"),
            "confidence": self._coerce_confidence(parsed.get("confidence")),
            "reason": str(parsed.get("reason") or "").strip(),
            "candidates_considered": candidates[: self.max_candidates],
        }
        return {
            "success": True,
            "answer": json.dumps(answer_payload, ensure_ascii=False, indent=2),
            "data": {
                "route": "source_select",
                "sources": [selected.get("raw_source") or {}],
                "metadata": {
                    **answer_payload,
                    "candidate_count": len(candidates),
                    "llm_calls": 1,
                },
            },
        }

    def _collect_url_candidates(self, previous_records: list[dict[str, Any]]) -> list[dict[str, Any]]:
        candidates: list[dict[str, Any]] = []
        seen_urls = set()
        for record in previous_records:
            for source in record.get("sources") or []:
                if not isinstance(source, dict):
                    continue
                url = source.get("url") or source.get("source_path")
                if not isinstance(url, str) or not url.startswith(("http://", "https://")):
                    continue
                if url in seen_urls:
                    continue
                seen_urls.add(url)
                candidates.append(
                    {
                        "source_index": len(candidates) + 1,
                        "title": source.get("title") or source.get("file_name") or url,
                        "url": url,
                        "snippet": str(source.get("text") or "")[:800],
                        "source_type": source.get("source_type"),
                        "raw_source": source,
                    }
                )
        return candidates

    @staticmethod
    def _parse_json(text: str) -> dict[str, Any]:
        match = re.search(r"```(?:json)?\s*(.*?)```", text, re.DOTALL)
        if match:
            text = match.group(1).strip()
        else:
            start = text.find("{")
            end = text.rfind("}")
            if start >= 0 and end >= start:
                text = text[start : end + 1]
        parsed = json.loads(text)
        if not isinstance(parsed, dict):
            raise ValueError("source_select output must be a JSON object")
        return parsed

    @staticmethod
    def _coerce_confidence(value: Any) -> float:
        try:
            number = float(value)
        except (TypeError, ValueError):
            return 0.0
        return max(0.0, min(1.0, number))


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


class WebSearchTool:
    def __init__(self):
        self.name = "web_search"
        self.provider = os.getenv("WEB_SEARCH_PROVIDER", "").strip().lower()
        self.api_key = os.getenv("WEB_SEARCH_API_KEY", "").strip() or os.getenv("EXA_API_KEY", "").strip()
        self.max_results = int(os.getenv("WEB_SEARCH_MAX_RESULTS", "5"))
        self.timeout_seconds = float(os.getenv("WEB_SEARCH_TIMEOUT_SECONDS", "10"))
        self.exa_mcp_endpoint = os.getenv(
            "WEB_SEARCH_EXA_MCP_ENDPOINT",
            _MCP_URL,
        ).strip()
        self.exa_mcp_tool_name = os.getenv("WEB_SEARCH_EXA_TOOL_NAME", "web_search_exa").strip()

    def structured_search(self, payload: dict[str, Any]) -> dict[str, Any]:
        query = str(payload.get("query") or payload.get("description") or "").strip()
        if not query and payload.get("entities"):
            query = " ".join(str(item) for item in payload.get("entities") or [])
        if not query:
            return {"success": False, "answer": "Web Search 查询为空。", "error": "empty web query"}

        if self.provider == "mock":
            return self._mock_search(query)
        if self.provider == "tavily":
            if not self.api_key or self.api_key.startswith("your_"):
                return self._unconfigured_result(query, "WEB_SEARCH_API_KEY is missing")
            return self._tavily_search(query)
        if self.provider in {"exa_mcp", "exa"}:
            if "{api_key}" in self.exa_mcp_endpoint and (not self.api_key or self.api_key.startswith("your_")):
                return self._unconfigured_result(query, "WEB_SEARCH_API_KEY or EXA_API_KEY is missing")
            return self._exa_mcp_search(query)
        return self._unconfigured_result(query, "WEB_SEARCH_PROVIDER is not configured")

    def _tavily_search(self, query: str) -> dict[str, Any]:
        payload = {
            "api_key": self.api_key,
            "query": query,
            "max_results": self.max_results,
            "include_answer": True,
            "search_depth": os.getenv("WEB_SEARCH_DEPTH", "basic"),
        }
        endpoint = os.getenv("WEB_SEARCH_ENDPOINT", "https://api.tavily.com/search")
        try:
            response = httpx.post(endpoint, json=payload, timeout=self.timeout_seconds)
            response.raise_for_status()
            data = response.json()
        except Exception as exc:
            return {
                "success": False,
                "answer": f"Web Search 调用失败：{exc}",
                "error": str(exc),
                "data": {
                    "route": "web_search",
                    "sources": [],
                    "metadata": {
                        "provider": self.provider,
                        "configured": True,
                        "query": query,
                        "max_results": self.max_results,
                    },
                },
            }

        results = data.get("results") if isinstance(data, dict) else []
        sources = self._sources_from_results(results if isinstance(results, list) else [])
        answer = self._format_answer(query, data.get("answer") if isinstance(data, dict) else None, sources)
        return {
            "success": True,
            "answer": answer,
            "data": {
                "route": "web_search",
                "sources": sources,
                "metadata": {
                    "provider": self.provider,
                    "configured": True,
                    "query": query,
                    "max_results": self.max_results,
                    "result_count": len(sources),
                },
            },
        }

    def _exa_mcp_search(self, query: str) -> dict[str, Any]:
        endpoint = self._build_exa_mcp_endpoint()
        payload = {
            "jsonrpc": "2.0",
            "id": "web_search_exa",
            "method": "tools/call",
            "params": {
                "name": self.exa_mcp_tool_name,
                "arguments": {
                    "query": query,
                    "numResults": self.max_results,
                },
            },
        }
        headers = {
            "accept": "application/json, text/event-stream",
            "content-type": "application/json",
        }
        if self.api_key and not self.api_key.startswith("your_"):
            headers["authorization"] = f"Bearer {self.api_key}"
        try:
            response = httpx.post(endpoint, json=payload, headers=headers, timeout=self.timeout_seconds)
            response.raise_for_status()
            messages = self._parse_exa_mcp_response(response)
            answer, sources = self._extract_exa_mcp_answer_sources(query, messages)
        except Exception as exc:
            return {
                "success": False,
                "answer": f"Web Search call failed: {exc}",
                "error": str(exc),
                "data": {
                    "route": "web_search",
                    "sources": [],
                    "metadata": {
                        "provider": self.provider,
                        "configured": True,
                        "query": query,
                        "max_results": self.max_results,
                        "endpoint": self._safe_endpoint(endpoint),
                        "tool_name": self.exa_mcp_tool_name,
                    },
                },
            }

        return {
            "success": True,
            "answer": answer,
            "data": {
                "route": "web_search",
                "sources": sources,
                "metadata": {
                    "provider": self.provider,
                    "configured": True,
                    "query": query,
                    "max_results": self.max_results,
                    "endpoint": self._safe_endpoint(endpoint),
                    "tool_name": self.exa_mcp_tool_name,
                    "event_count": len(messages),
                    "result_count": len(sources),
                },
            },
        }

    def _mock_search(self, query: str) -> dict[str, Any]:
        raw = os.getenv("WEB_SEARCH_MOCK_RESULTS", "[]")
        try:
            parsed = json.loads(raw)
        except Exception:
            parsed = []
        sources = self._sources_from_results(parsed if isinstance(parsed, list) else [])
        answer = self._format_answer(query, None, sources)
        return {
            "success": True,
            "answer": answer,
            "data": {
                "route": "web_search",
                "sources": sources,
                "metadata": {
                    "provider": "mock",
                    "configured": True,
                    "query": query,
                    "max_results": self.max_results,
                    "result_count": len(sources),
                },
            },
        }

    def _unconfigured_result(self, query: str, reason: str) -> dict[str, Any]:
        return {
            "success": False,
            "answer": WEB_SEARCH_UNCONFIGURED_MESSAGE,
            "error": reason,
            "data": {
                "route": "web_search",
                "sources": [],
                "metadata": {
                    "provider": self.provider or None,
                    "configured": False,
                    "query": query,
                    "reason": reason,
                },
            },
        }

    def _build_exa_mcp_endpoint(self) -> str:
        endpoint = self.exa_mcp_endpoint
        if "{api_key}" in endpoint:
            return endpoint.replace("{api_key}", self.api_key)
        return endpoint

    def _safe_endpoint(self, endpoint: str) -> str:
        if self.api_key:
            return endpoint.replace(self.api_key, "***")
        return endpoint

    def _parse_exa_mcp_response(self, response: Any) -> list[dict[str, Any]]:
        headers = getattr(response, "headers", {}) or {}
        content_type = str(headers.get("content-type", "")).lower()
        text = str(getattr(response, "text", "") or "")
        if "text/event-stream" in content_type or any(
            line.strip().startswith("data:") for line in text.splitlines()
        ):
            return self._parse_sse_data_events(text)
        try:
            data = response.json()
        except Exception:
            try:
                data = json.loads(text)
            except Exception:
                data = {"raw_text": text}
        return [data if isinstance(data, dict) else {"data": data}]

    def _parse_sse_data_events(self, text: str) -> list[dict[str, Any]]:
        events: list[dict[str, Any]] = []
        data_lines: list[str] = []

        def flush_event() -> None:
            if not data_lines:
                return
            raw = "\n".join(data_lines).strip()
            data_lines.clear()
            if not raw or raw == "[DONE]":
                return
            try:
                parsed = json.loads(raw)
            except Exception:
                parsed = {"raw_text": raw}
            events.append(parsed if isinstance(parsed, dict) else {"data": parsed})

        for line in text.splitlines():
            stripped = line.strip()
            if not stripped:
                flush_event()
                continue
            if stripped.startswith("data:"):
                data_lines.append(stripped[5:].strip())
        flush_event()
        return events

    def _extract_exa_mcp_answer_sources(
        self,
        query: str,
        messages: list[dict[str, Any]],
    ) -> tuple[str, list[dict[str, Any]]]:
        text_parts: list[str] = []
        sources: list[dict[str, Any]] = []
        for message in messages:
            text_parts.extend(self._extract_text_parts(message))
            sources.extend(self._extract_sources_from_exa_payload(message))

        sources = self._dedupe_web_sources(sources)
        answer = "\n\n".join(part for part in text_parts if part).strip()
        if (not answer or self._looks_like_json(answer)) and sources:
            answer = self._format_answer(query, None, sources)
        if not answer:
            answer = self._format_answer(query, None, sources)
        return answer, sources

    def _extract_text_parts(self, payload: Any) -> list[str]:
        parts: list[str] = []
        if isinstance(payload, list):
            for item in payload:
                parts.extend(self._extract_text_parts(item))
            return parts
        if not isinstance(payload, dict):
            return parts

        raw_text = payload.get("raw_text")
        if isinstance(raw_text, str):
            parts.append(raw_text)

        text = payload.get("text")
        if isinstance(text, str):
            parts.append(text)

        for key in ("result", "data", "output"):
            value = payload.get(key)
            if isinstance(value, (dict, list)):
                parts.extend(self._extract_text_parts(value))

        content = payload.get("content")
        if isinstance(content, list):
            for item in content:
                if isinstance(item, dict) and isinstance(item.get("text"), str):
                    parts.append(item["text"])
                else:
                    parts.extend(self._extract_text_parts(item))
        return parts

    def _extract_sources_from_exa_payload(self, payload: Any) -> list[dict[str, Any]]:
        if isinstance(payload, list):
            direct_results = [item for item in payload if isinstance(item, dict)]
            if direct_results and any(item.get("url") or item.get("link") for item in direct_results):
                return self._sources_from_results(direct_results)
            sources: list[dict[str, Any]] = []
            for item in payload:
                sources.extend(self._extract_sources_from_exa_payload(item))
            return sources

        if not isinstance(payload, dict):
            return []

        sources: list[dict[str, Any]] = []
        for key in ("results", "searchResults"):
            results = payload.get(key)
            if isinstance(results, list):
                sources.extend(self._sources_from_results(results))

        text = payload.get("text") or payload.get("raw_text")
        if isinstance(text, str):
            sources.extend(self._extract_sources_from_exa_text(text))

        for key in ("result", "structuredContent", "data", "output", "content"):
            value = payload.get(key)
            if isinstance(value, (dict, list)):
                sources.extend(self._extract_sources_from_exa_payload(value))
        return sources

    def _extract_sources_from_exa_text(self, text: str) -> list[dict[str, Any]]:
        stripped = text.strip()
        if self._looks_like_json(stripped):
            try:
                parsed = json.loads(stripped)
            except Exception:
                return []
            return self._extract_sources_from_exa_payload(parsed)
        return self._extract_sources_from_plain_exa_text(stripped)

    def _extract_sources_from_plain_exa_text(self, text: str) -> list[dict[str, Any]]:
        results: list[dict[str, Any]] = []
        current: dict[str, Any] | None = None
        collecting_highlights = False

        def flush_current() -> None:
            nonlocal current, collecting_highlights
            if current and current.get("url"):
                content = current.get("content")
                if isinstance(content, list):
                    current["content"] = "\n".join(item for item in content if item).strip() or None
                results.append(current)
            current = None
            collecting_highlights = False

        for raw_line in text.splitlines():
            line = raw_line.strip()
            if not line:
                collecting_highlights = False
                continue

            if line.startswith("Title:"):
                flush_current()
                current = {"title": line.removeprefix("Title:").strip()}
                continue
            if current is None:
                continue

            if line.startswith("URL:"):
                current["url"] = line.removeprefix("URL:").strip().split()[0]
                collecting_highlights = False
                continue
            if line.startswith("Published:"):
                published = line.removeprefix("Published:").strip()
                current["published_at"] = None if published.upper() == "N/A" else published
                collecting_highlights = False
                continue
            if line.startswith("Highlights:"):
                highlight = line.removeprefix("Highlights:").strip()
                current.setdefault("content", [])
                if highlight:
                    current["content"].append(highlight)
                collecting_highlights = True
                continue
            if collecting_highlights:
                current.setdefault("content", [])
                current["content"].append(line)

        flush_current()

        if not results:
            for match in re.finditer(r"(?m)^URL:\s*(https?://\S+)", text):
                results.append({"url": match.group(1)})
        return self._sources_from_results(results)

    @staticmethod
    def _looks_like_json(text: str) -> bool:
        stripped = text.strip()
        return (stripped.startswith("{") and stripped.endswith("}")) or (
            stripped.startswith("[") and stripped.endswith("]")
        )

    @staticmethod
    def _dedupe_web_sources(sources: list[dict[str, Any]]) -> list[dict[str, Any]]:
        deduped: list[dict[str, Any]] = []
        seen = set()
        for source in sources:
            key = source.get("url") or source.get("title") or source.get("text")
            if key in seen:
                continue
            seen.add(key)
            deduped.append(source)
        return deduped

    def _sources_from_results(self, results: list[dict[str, Any]]) -> list[dict[str, Any]]:
        sources = []
        for index, item in enumerate(results[: self.max_results], start=1):
            if not isinstance(item, dict):
                continue
            url = item.get("url") or item.get("link")
            title = item.get("title") or item.get("name") or url
            content = item.get("content") or item.get("snippet") or item.get("description")
            sources.append(
                {
                    "doc_id": f"web:{index}",
                    "chunk_id": f"web:{index}",
                    "title": str(title) if title else None,
                    "url": str(url) if url else None,
                    "source_path": str(url) if url else None,
                    "file_name": str(title) if title else None,
                    "published_at": item.get("published_date") or item.get("published_at"),
                    "source_type": "web",
                    "text": str(content) if content else None,
                    "score": item.get("score"),
                }
            )
        return sources

    @staticmethod
    def _format_answer(query: str, answer: str | None, sources: list[dict[str, Any]]) -> str:
        lines = [f"Web Search 查询：{query}"]
        if answer:
            lines.extend(["", str(answer)])
        if sources:
            lines.extend(["", "网页来源："])
            for index, source in enumerate(sources, start=1):
                label = source.get("title") or source.get("url") or f"结果 {index}"
                url = source.get("url") or ""
                lines.append(f"{index}. {label} {url}".strip())
        else:
            lines.extend(["", "未返回网页结果。"])
        return "\n".join(lines)


class WeatherQueryTool:
    def __init__(self):
        self.name = "weather_query"
        self.provider = (
            os.getenv("WEATHER_QUERY_PROVIDER")
            or os.getenv("WEATHER_PROVIDER")
            or "open_meteo"
        ).strip().lower()
        self.timeout_seconds = float(os.getenv("WEATHER_QUERY_TIMEOUT_SECONDS", "10"))
        self.max_forecast_days = int(os.getenv("WEATHER_QUERY_MAX_FORECAST_DAYS", "16"))
        self.default_location = os.getenv("WEATHER_QUERY_DEFAULT_LOCATION", "桂林").strip() or "桂林"
        self.open_meteo_geocoding_endpoint = os.getenv(
            "OPEN_METEO_GEOCODING_ENDPOINT",
            "https://geocoding-api.open-meteo.com/v1/search",
        ).strip()
        self.open_meteo_forecast_endpoint = os.getenv(
            "OPEN_METEO_FORECAST_ENDPOINT",
            "https://api.open-meteo.com/v1/forecast",
        ).strip()

    def structured_search(self, payload: dict[str, Any]) -> dict[str, Any]:
        query = str(payload.get("query") or payload.get("description") or "").strip()
        location = self._resolve_location(payload, query)
        start_date, end_date, date_error = self._resolve_date_range(payload, query)
        if date_error:
            return self._error_result(
                location=location,
                start_date=None,
                end_date=None,
                reason="invalid_date_range",
                message=date_error,
            )

        assert start_date is not None and end_date is not None
        availability = self._forecast_availability(start_date, end_date)
        if availability:
            return self._unavailable_forecast_result(
                location=location,
                start_date=start_date,
                end_date=end_date,
                reason=availability["reason"],
                message=availability["message"],
            )

        if self.provider == "mock":
            return self._mock_weather(location, start_date, end_date, query)
        if self.provider in {"open_meteo", "open-meteo", "openmeteo"}:
            return self._open_meteo_weather(location, start_date, end_date, query)

        return self._error_result(
            location=location,
            start_date=start_date,
            end_date=end_date,
            reason="unsupported_provider",
            message=f"Unsupported weather provider: {self.provider}",
            metadata={"configured": False, "provider": self.provider or None},
        )

    def _open_meteo_weather(
        self,
        location: str,
        start_date: date,
        end_date: date,
        query: str,
    ) -> dict[str, Any]:
        try:
            resolved = self._geocode_open_meteo(location)
            forecast = self._forecast_open_meteo(resolved, start_date, end_date)
        except Exception as exc:
            return self._error_result(
                location=location,
                start_date=start_date,
                end_date=end_date,
                reason="provider_call_failed",
                message=str(exc),
                metadata={"configured": True, "provider": "open_meteo"},
            )

        daily = self._daily_from_open_meteo(forecast)
        return self._success_result(
            location=location,
            resolved_location=resolved,
            start_date=start_date,
            end_date=end_date,
            daily=daily,
            query=query,
            provider="open_meteo",
            metadata={
                "configured": True,
                "geocoding_endpoint": self.open_meteo_geocoding_endpoint,
                "forecast_endpoint": self.open_meteo_forecast_endpoint,
            },
        )

    def _geocode_open_meteo(self, location: str) -> dict[str, Any]:
        response = httpx.get(
            self.open_meteo_geocoding_endpoint,
            params={
                "name": location,
                "count": 1,
                "language": "zh",
                "format": "json",
            },
            timeout=self.timeout_seconds,
        )
        response.raise_for_status()
        data = response.json()
        results = data.get("results") if isinstance(data, dict) else None
        if not results:
            raise ValueError(f"Weather location not found: {location}")
        first = results[0]
        return {
            "name": first.get("name") or location,
            "country": first.get("country"),
            "admin1": first.get("admin1"),
            "latitude": first.get("latitude"),
            "longitude": first.get("longitude"),
            "timezone": first.get("timezone"),
        }

    def _forecast_open_meteo(
        self,
        resolved_location: dict[str, Any],
        start_date: date,
        end_date: date,
    ) -> dict[str, Any]:
        latitude = resolved_location.get("latitude")
        longitude = resolved_location.get("longitude")
        if latitude is None or longitude is None:
            raise ValueError("Open-Meteo geocoding result did not include coordinates")

        response = httpx.get(
            self.open_meteo_forecast_endpoint,
            params={
                "latitude": latitude,
                "longitude": longitude,
                "daily": ",".join(
                    [
                        "weather_code",
                        "temperature_2m_max",
                        "temperature_2m_min",
                        "precipitation_probability_max",
                        "precipitation_sum",
                        "wind_speed_10m_max",
                    ]
                ),
                "timezone": "auto",
                "start_date": start_date.isoformat(),
                "end_date": end_date.isoformat(),
            },
            timeout=self.timeout_seconds,
        )
        response.raise_for_status()
        data = response.json()
        if not isinstance(data, dict) or not isinstance(data.get("daily"), dict):
            raise ValueError("Open-Meteo forecast response did not include daily data")
        return data

    def _daily_from_open_meteo(self, forecast: dict[str, Any]) -> list[dict[str, Any]]:
        daily = forecast.get("daily") or {}
        dates = daily.get("time") or []
        rows: list[dict[str, Any]] = []
        for index, raw_date in enumerate(dates):
            weather_code = self._daily_value(daily, "weather_code", index)
            temp_max = self._daily_value(daily, "temperature_2m_max", index)
            temp_min = self._daily_value(daily, "temperature_2m_min", index)
            precipitation_probability = self._daily_value(daily, "precipitation_probability_max", index)
            precipitation_sum = self._daily_value(daily, "precipitation_sum", index)
            wind_speed = self._daily_value(daily, "wind_speed_10m_max", index)
            rows.append(
                {
                    "date": str(raw_date),
                    "condition": self._weather_code_label(weather_code),
                    "weather_code": weather_code,
                    "temperature_min_c": temp_min,
                    "temperature_max_c": temp_max,
                    "precipitation_probability_max": precipitation_probability,
                    "precipitation_sum_mm": precipitation_sum,
                    "wind_speed_10m_max_kmh": wind_speed,
                    "travel_impact": self._travel_impact(
                        weather_code=weather_code,
                        temp_max=temp_max,
                        precipitation_probability=precipitation_probability,
                        precipitation_sum=precipitation_sum,
                        wind_speed=wind_speed,
                    ),
                }
            )
        return rows

    def _mock_weather(self, location: str, start_date: date, end_date: date, query: str) -> dict[str, Any]:
        raw = os.getenv("WEATHER_QUERY_MOCK_RESULT")
        if raw:
            try:
                parsed = json.loads(raw)
            except Exception:
                parsed = {}
            if isinstance(parsed, dict):
                daily = parsed.get("daily") if isinstance(parsed.get("daily"), list) else []
                resolved = parsed.get("resolved_location") if isinstance(parsed.get("resolved_location"), dict) else {}
                return self._success_result(
                    location=parsed.get("location") or location,
                    resolved_location=resolved or {"name": location},
                    start_date=start_date,
                    end_date=end_date,
                    daily=daily,
                    query=query,
                    provider="mock",
                    metadata={"configured": True},
                    planning_hint=parsed.get("planning_hint"),
                )

        daily = []
        current = start_date
        while current <= end_date:
            daily.append(
                {
                    "date": current.isoformat(),
                    "condition": "clear",
                    "weather_code": 0,
                    "temperature_min_c": 23,
                    "temperature_max_c": 30,
                    "precipitation_probability_max": 10,
                    "precipitation_sum_mm": 0,
                    "wind_speed_10m_max_kmh": 10,
                    "travel_impact": "outdoor_friendly",
                }
            )
            current += timedelta(days=1)
        return self._success_result(
            location=location,
            resolved_location={"name": location},
            start_date=start_date,
            end_date=end_date,
            daily=daily,
            query=query,
            provider="mock",
            metadata={"configured": True},
        )

    def _success_result(
        self,
        *,
        location: str,
        resolved_location: dict[str, Any],
        start_date: date,
        end_date: date,
        daily: list[dict[str, Any]],
        query: str,
        provider: str,
        metadata: dict[str, Any] | None = None,
        planning_hint: str | None = None,
    ) -> dict[str, Any]:
        fetched_at = datetime.now(timezone.utc).isoformat()
        hint = planning_hint or self._planning_hint(daily)
        response_data = {
            "location": location,
            "resolved_location": resolved_location,
            "date_range": {
                "start_date": start_date.isoformat(),
                "end_date": end_date.isoformat(),
            },
            "provider": provider,
            "forecast_available": True,
            "daily": daily,
            "planning_hint": hint,
            "fetched_at": fetched_at,
        }
        answer = json.dumps(response_data, ensure_ascii=False, indent=2)
        source_text = self._source_text(response_data)
        return {
            "success": True,
            "answer": answer,
            "data": {
                "route": "weather_query",
                "sources": [
                    {
                        "doc_id": f"weather:{provider}:{location}:{start_date.isoformat()}:{end_date.isoformat()}",
                        "chunk_id": f"weather:{provider}:{location}:{start_date.isoformat()}:{end_date.isoformat()}",
                        "title": f"Weather forecast for {location}",
                        "url": "https://open-meteo.com/" if provider == "open_meteo" else None,
                        "source_path": provider,
                        "file_name": f"{provider} weather forecast",
                        "source_type": "weather",
                        "text": source_text,
                        "score": None,
                    }
                ],
                "metadata": {
                    "provider": provider,
                    "configured": True,
                    "query": query,
                    "location": location,
                    "start_date": start_date.isoformat(),
                    "end_date": end_date.isoformat(),
                    "forecast_available": True,
                    "result_count": len(daily),
                    "fetched_at": fetched_at,
                    **(metadata or {}),
                },
            },
        }

    def _unavailable_forecast_result(
        self,
        *,
        location: str,
        start_date: date,
        end_date: date,
        reason: str,
        message: str,
    ) -> dict[str, Any]:
        response_data = {
            "location": location,
            "date_range": {
                "start_date": start_date.isoformat(),
                "end_date": end_date.isoformat(),
            },
            "provider": self.provider,
            "forecast_available": False,
            "daily": [],
            "planning_hint": message,
            "reason": reason,
            "fetched_at": datetime.now(timezone.utc).isoformat(),
        }
        return {
            "success": True,
            "answer": json.dumps(response_data, ensure_ascii=False, indent=2),
            "data": {
                "route": "weather_query",
                "sources": [],
                "metadata": {
                    "provider": self.provider,
                    "configured": True,
                    "location": location,
                    "start_date": start_date.isoformat(),
                    "end_date": end_date.isoformat(),
                    "forecast_available": False,
                    "reason": reason,
                },
            },
        }

    def _error_result(
        self,
        *,
        location: str | None,
        start_date: date | None,
        end_date: date | None,
        reason: str,
        message: str,
        metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        payload = {
            "location": location,
            "date_range": {
                "start_date": start_date.isoformat() if start_date else None,
                "end_date": end_date.isoformat() if end_date else None,
            },
            "error": reason,
            "message": message,
        }
        return {
            "success": False,
            "answer": json.dumps(payload, ensure_ascii=False, indent=2),
            "error": message,
            "data": {
                "route": "weather_query",
                "sources": [],
                "metadata": {
                    "provider": self.provider or None,
                    "configured": True,
                    "location": location,
                    "reason": reason,
                    **(metadata or {}),
                },
            },
        }

    def _resolve_location(self, payload: dict[str, Any], query: str) -> str:
        for key in ("location", "city", "place"):
            value = str(payload.get(key) or "").strip()
            if value:
                return value
        entities = payload.get("entities")
        if isinstance(entities, list):
            for entity in entities:
                text = str(entity).strip()
                if text:
                    return text

        text = f"{query} {payload.get('description') or ''}"
        known_locations = [
            "阳朔",
            "龙胜",
            "桂林",
            "漓江",
            "遇龙河",
            "两江四湖",
            "象鼻山",
            "靖江王府",
        ]
        for known in known_locations:
            if known in text:
                if known in {"漓江", "遇龙河", "两江四湖", "象鼻山", "靖江王府"}:
                    return "桂林"
                return known
        return self.default_location

    def _resolve_date_range(
        self,
        payload: dict[str, Any],
        query: str,
    ) -> tuple[date | None, date | None, str | None]:
        start = self._parse_date(payload.get("start_date"))
        end = self._parse_date(payload.get("end_date"))
        text = f"{query} {payload.get('description') or ''}"

        if not start:
            dates = self._extract_dates_from_text(text)
            if dates:
                start = dates[0]
                if len(dates) > 1:
                    end = dates[1]

        if not start:
            start = self._relative_date_from_text(text)

        duration_days = self._duration_days_from_text(text)
        if start and not end:
            end = start + timedelta(days=max(duration_days, 1) - 1)
        if start and end and end < start:
            return None, None, "end_date must be greater than or equal to start_date"
        if not start or not end:
            return None, None, "weather_query requires a travel date or date range"
        return start, end, None

    @staticmethod
    def _parse_date(value: Any) -> date | None:
        if isinstance(value, date) and not isinstance(value, datetime):
            return value
        if isinstance(value, datetime):
            return value.date()
        text = str(value or "").strip()
        if not text:
            return None
        for fmt in ("%Y-%m-%d", "%Y/%m/%d"):
            try:
                return datetime.strptime(text, fmt).date()
            except ValueError:
                continue
        match = re.search(r"(20\d{2})年(\d{1,2})月(\d{1,2})日?", text)
        if match:
            return date(int(match.group(1)), int(match.group(2)), int(match.group(3)))
        return None

    def _extract_dates_from_text(self, text: str) -> list[date]:
        dates: list[date] = []
        for match in re.finditer(r"(20\d{2})[-/.年](\d{1,2})[-/.月](\d{1,2})日?", text):
            try:
                dates.append(date(int(match.group(1)), int(match.group(2)), int(match.group(3))))
            except ValueError:
                continue

        today = date.today()
        for match in re.finditer(r"(?<!\d)(\d{1,2})月(\d{1,2})日?", text):
            try:
                candidate = date(today.year, int(match.group(1)), int(match.group(2)))
                if candidate < today:
                    candidate = date(today.year + 1, int(match.group(1)), int(match.group(2)))
                if candidate not in dates:
                    dates.append(candidate)
            except ValueError:
                continue
        return dates

    @staticmethod
    def _relative_date_from_text(text: str) -> date | None:
        today = date.today()
        if "今天" in text:
            return today
        if "明天" in text:
            return today + timedelta(days=1)
        if "后天" in text:
            return today + timedelta(days=2)
        if "下周末" in text or "周末" in text:
            days_until_saturday = (5 - today.weekday()) % 7
            if days_until_saturday == 0:
                days_until_saturday = 7
            if "下周末" in text:
                days_until_saturday += 7
            return today + timedelta(days=days_until_saturday)
        if "下周" in text:
            days_until_next_monday = (7 - today.weekday()) % 7
            if days_until_next_monday == 0:
                days_until_next_monday = 7
            return today + timedelta(days=days_until_next_monday)
        return None

    def _duration_days_from_text(self, text: str) -> int:
        match = re.search(r"([一二两三四五六七八九十\d]+)\s*(?:天|日)游", text)
        if not match:
            match = re.search(r"([一二两三四五六七八九十\d]+)\s*天", text)
        if not match:
            return 1
        raw = match.group(1)
        if raw.isdigit():
            return max(int(raw), 1)
        return self._simple_chinese_number(raw) or 1

    @staticmethod
    def _simple_chinese_number(text: str) -> int | None:
        mapping = {
            "一": 1,
            "二": 2,
            "两": 2,
            "三": 3,
            "四": 4,
            "五": 5,
            "六": 6,
            "七": 7,
            "八": 8,
            "九": 9,
            "十": 10,
        }
        if text in mapping:
            return mapping[text]
        if text.startswith("十") and len(text) == 2:
            return 10 + mapping.get(text[1], 0)
        if len(text) == 2 and text.endswith("十"):
            return mapping.get(text[0], 0) * 10
        if len(text) == 3 and text[1] == "十":
            return mapping.get(text[0], 0) * 10 + mapping.get(text[2], 0)
        return None

    def _forecast_availability(self, start_date: date, end_date: date) -> dict[str, str] | None:
        today = date.today()
        max_date = today + timedelta(days=max(self.max_forecast_days, 1) - 1)
        if start_date < today:
            return {
                "reason": "past_date_unsupported",
                "message": "Weather forecast provider only supports current and future forecast dates.",
            }
        if end_date > max_date:
            return {
                "reason": "forecast_range_exceeded",
                "message": (
                    f"Weather forecast is only available through {max_date.isoformat()} "
                    f"for this provider. Use seasonal climate guidance for farther dates."
                ),
            }
        return None

    @staticmethod
    def _daily_value(daily: dict[str, Any], key: str, index: int) -> Any:
        values = daily.get(key)
        if isinstance(values, list) and index < len(values):
            return values[index]
        return None

    @staticmethod
    def _weather_code_label(code: Any) -> str:
        try:
            value = int(code)
        except (TypeError, ValueError):
            return "unknown"
        if value == 0:
            return "clear"
        if value in {1, 2, 3}:
            return "cloudy"
        if value in {45, 48}:
            return "fog"
        if value in {51, 53, 55, 56, 57}:
            return "drizzle"
        if value in {61, 63, 65, 66, 67, 80, 81, 82}:
            return "rain"
        if value in {71, 73, 75, 77, 85, 86}:
            return "snow"
        if value in {95, 96, 99}:
            return "thunderstorm"
        return "unknown"

    def _travel_impact(
        self,
        *,
        weather_code: Any,
        temp_max: Any,
        precipitation_probability: Any,
        precipitation_sum: Any,
        wind_speed: Any,
    ) -> str:
        condition = self._weather_code_label(weather_code)
        precip_prob = self._safe_float(precipitation_probability)
        precip_sum = self._safe_float(precipitation_sum)
        max_temp = self._safe_float(temp_max)
        wind = self._safe_float(wind_speed)

        if condition in {"rain", "thunderstorm"} or precip_prob >= 60 or precip_sum >= 5:
            return "rain_sensitive: prefer indoor or covered attractions; avoid rafting, hiking, and exposed viewpoints."
        if max_temp >= 33:
            return "hot: schedule outdoor activities in morning/evening and keep noon for indoor stops."
        if wind >= 38:
            return "windy: check boats, cableways, and exposed mountain routes before departure."
        if condition in {"clear", "cloudy"}:
            return "outdoor_friendly"
        return "neutral"

    @staticmethod
    def _safe_float(value: Any) -> float:
        try:
            if value is None:
                return 0.0
            return float(value)
        except (TypeError, ValueError):
            return 0.0

    @staticmethod
    def _planning_hint(daily: list[dict[str, Any]]) -> str:
        impacts = " ".join(str(item.get("travel_impact") or "") for item in daily)
        if "rain_sensitive" in impacts:
            return "Rain is likely during the trip. Add indoor alternatives and avoid long outdoor or water-based activities on rainy days."
        if "hot" in impacts:
            return "High temperature is expected. Put outdoor attractions in early morning or evening and reserve noon for indoor/rest stops."
        if "windy" in impacts:
            return "Wind may affect boats, cableways, or exposed viewpoints. Check operations before fixing the route."
        return "Weather is generally suitable for regular outdoor sightseeing."

    @staticmethod
    def _source_text(response_data: dict[str, Any]) -> str:
        lines = [
            f"provider: {response_data.get('provider')}",
            f"location: {response_data.get('location')}",
            f"date_range: {response_data.get('date_range')}",
            f"planning_hint: {response_data.get('planning_hint')}",
        ]
        for item in response_data.get("daily") or []:
            lines.append(
                " | ".join(
                    [
                        f"date={item.get('date')}",
                        f"condition={item.get('condition')}",
                        f"temp={item.get('temperature_min_c')}-{item.get('temperature_max_c')}C",
                        f"precip_prob={item.get('precipitation_probability_max')}",
                        f"impact={item.get('travel_impact')}",
                    ]
                )
            )
        return "\n".join(lines)


class AmapRouteProvider:
    GEOCODE_ENDPOINT = "https://restapi.amap.com/v3/geocode/geo"
    DIRECTION_ENDPOINTS = {
        "walk": "https://restapi.amap.com/v3/direction/walking",
        "walking": "https://restapi.amap.com/v3/direction/walking",
        "taxi": "https://restapi.amap.com/v3/direction/driving",
        "driving": "https://restapi.amap.com/v3/direction/driving",
        "drive": "https://restapi.amap.com/v3/direction/driving",
        "auto": "https://restapi.amap.com/v3/direction/driving",
        "transit": "https://restapi.amap.com/v3/direction/transit/integrated",
    }

    def __init__(self, api_key: str, *, city: str | None = None, timeout_seconds: float = 10.0):
        self.api_key = api_key.strip()
        self.city = city.strip() if city else None
        self.timeout_seconds = timeout_seconds
        logging.getLogger("httpx").setLevel(logging.WARNING)
        self._geocode_cache: dict[tuple[str, str | None], dict[str, Any]] = {}
        self._route_cache: dict[tuple[str, str, str], dict[str, Any]] = {}

    @property
    def configured(self) -> bool:
        return bool(self.api_key)

    def geocode(self, name: str) -> dict[str, Any] | None:
        if not self.configured:
            return None
        cache_key = (name, self.city)
        if cache_key in self._geocode_cache:
            return dict(self._geocode_cache[cache_key])

        params = {
            "key": self.api_key,
            "address": name,
            "output": "JSON",
        }
        if self.city:
            params["city"] = self.city

        response = httpx.get(self.GEOCODE_ENDPOINT, params=params, timeout=self.timeout_seconds)
        response.raise_for_status()
        data = response.json()
        if str(data.get("status")) != "1":
            raise ValueError(f"Amap geocode failed: {data.get('info') or data.get('infocode')}")
        geocodes = data.get("geocodes") if isinstance(data, dict) else None
        if not geocodes:
            return None

        first = geocodes[0]
        lng_lat = str(first.get("location") or "").split(",")
        if len(lng_lat) != 2:
            return None
        lng = float(lng_lat[0])
        lat = float(lng_lat[1])
        resolved = {
            "name": name,
            "lat": lat,
            "lng": lng,
            "formatted_address": first.get("formatted_address"),
            "geocode_level": first.get("level"),
            "coordinate_provider": "amap",
        }
        self._geocode_cache[cache_key] = dict(resolved)
        return resolved

    def route(self, start: dict[str, Any], end: dict[str, Any], mode: str) -> dict[str, Any]:
        if not self.configured:
            raise ValueError("AMAP_API_KEY is not configured")

        normalized_mode = self._normalize_route_mode(mode)
        cache_key = (
            f"{start['lng']:.6f},{start['lat']:.6f}",
            f"{end['lng']:.6f},{end['lat']:.6f}",
            normalized_mode,
        )
        if cache_key in self._route_cache:
            cached = dict(self._route_cache[cache_key])
            cached["from"] = start["name"]
            cached["to"] = end["name"]
            return cached

        endpoint = self.DIRECTION_ENDPOINTS[normalized_mode]
        params = {
            "key": self.api_key,
            "origin": cache_key[0],
            "destination": cache_key[1],
            "output": "JSON",
        }
        if normalized_mode == "transit" and self.city:
            params["city"] = self.city
            params["cityd"] = self.city

        response = httpx.get(endpoint, params=params, timeout=self.timeout_seconds)
        response.raise_for_status()
        data = response.json()
        if str(data.get("status")) != "1":
            raise ValueError(f"Amap route failed: {data.get('info') or data.get('infocode')}")

        distance_m, duration_sec = self._extract_route_metrics(data, normalized_mode)
        leg = {
            "from": start["name"],
            "to": end["name"],
            "mode": mode,
            "distance_m": max(1, int(distance_m)),
            "duration_min": max(1, int(math.ceil(float(duration_sec) / 60))),
            "provider": "amap",
        }
        self._route_cache[cache_key] = dict(leg)
        return leg

    @staticmethod
    def _normalize_route_mode(mode: str) -> str:
        if mode in {"walk", "walking"}:
            return "walk"
        if mode == "transit":
            return "transit"
        return "driving"

    @staticmethod
    def _extract_route_metrics(data: dict[str, Any], mode: str) -> tuple[float, float]:
        route = data.get("route") if isinstance(data, dict) else None
        if not isinstance(route, dict):
            raise ValueError("Amap route response missing route")
        if mode == "transit":
            transits = route.get("transits") or []
            if not transits:
                raise ValueError("Amap route response missing transits")
            first = transits[0]
            return float(first.get("distance") or 0), float(first.get("duration") or 0)

        paths = route.get("paths") or []
        if not paths:
            raise ValueError("Amap route response missing paths")
        first = paths[0]
        return float(first.get("distance") or 0), float(first.get("duration") or 0)


class MapRouteTool:
    LOCAL_COORDINATES = {
        "桂林站": (25.2610, 110.2866),
        "桂林北站": (25.3306, 110.3013),
        "桂林西站": (25.3300, 110.2015),
        "桂林两江国际机场": (25.2181, 110.0392),
        "象鼻山": (25.2742, 110.2950),
        "靖江王府": (25.2891, 110.2995),
        "独秀峰": (25.2898, 110.3000),
        "东西巷": (25.2848, 110.2989),
        "正阳步行街": (25.2784, 110.2961),
        "两江四湖": (25.2795, 110.2925),
        "日月双塔": (25.2747, 110.2947),
        "七星公园": (25.2798, 110.3152),
        "芦笛岩": (25.3168, 110.2718),
        "阳朔西街": (24.7785, 110.4969),
        "西街": (24.7785, 110.4969),
        "遇龙河": (24.7647, 110.4556),
        "十里画廊": (24.7489, 110.4831),
        "银子岩": (24.6546, 110.4618),
        "漓江": (25.1645, 110.4241),
        "兴坪古镇": (24.9197, 110.5269),
        "相公山": (24.9522, 110.4889),
        "龙脊梯田": (25.7595, 110.1230),
        "龙胜温泉": (25.8946, 110.1466),
    }
    DEFAULT_ROUTE_PLACES = {
        "阳朔": ["遇龙河", "十里画廊", "阳朔西街"],
        "龙胜": ["龙脊梯田", "龙胜温泉"],
        "桂林": ["象鼻山", "靖江王府", "东西巷", "两江四湖"],
        "市区": ["象鼻山", "靖江王府", "东西巷", "两江四湖"],
    }

    MODE_SPEED_KMH = {
        "walk": 4.5,
        "walking": 4.5,
        "taxi": 28.0,
        "driving": 30.0,
        "drive": 30.0,
        "transit": 18.0,
        "auto": 26.0,
    }

    def __init__(self):
        self.name = "map_route"
        self.provider = os.getenv("MAP_ROUTE_PROVIDER", "local").strip().lower()
        self.default_mode = os.getenv("MAP_ROUTE_DEFAULT_MODE", "taxi").strip().lower() or "taxi"
        self.default_origin = os.getenv("MAP_ROUTE_DEFAULT_ORIGIN", "").strip() or None
        self.default_visit_duration_min = int(os.getenv("MAP_ROUTE_DEFAULT_VISIT_DURATION_MIN", "90"))
        self.max_exact_places = int(os.getenv("MAP_ROUTE_MAX_EXACT_PLACES", "8"))
        self.road_factor = float(os.getenv("MAP_ROUTE_ROAD_FACTOR", "1.25"))
        self.amap_provider: AmapRouteProvider | None = None
        self._provider_warnings: list[str] = []
        if self.provider == "amap":
            self.amap_provider = AmapRouteProvider(
                os.getenv("AMAP_API_KEY", ""),
                city=os.getenv("AMAP_DEFAULT_CITY", "桂林"),
                timeout_seconds=float(os.getenv("AMAP_TIMEOUT_SECONDS", "10")),
            )

    def structured_search(self, payload: dict[str, Any]) -> dict[str, Any]:
        self._provider_warnings = []
        query = str(payload.get("query") or payload.get("description") or "").strip()
        mode = self._normalize_mode(payload.get("mode") or self.default_mode)
        origin = self._place_name(payload.get("origin") or payload.get("start") or self.default_origin)
        destination = self._place_name(payload.get("destination") or payload.get("end"))
        inferred_endpoints = self._infer_endpoints(query)
        origin = origin or inferred_endpoints.get("origin")
        destination = destination or inferred_endpoints.get("destination")
        if payload.get("return_to_origin") and origin and not destination:
            destination = origin

        places = self._resolve_places(payload, query)
        places = self._remove_origin_from_places(places, origin)
        if not places:
            return self._error_result(
                reason="missing_places",
                message="map_route requires at least one place in places, entities, or query text.",
                query=query,
            )

        points, missing = self._resolve_points(origin, destination, places)
        if missing:
            return self._error_result(
                reason="missing_coordinates",
                message=f"Missing coordinates for: {', '.join(missing)}",
                query=query,
                metadata={"missing_places": missing},
            )

        start_min = self._parse_time(payload.get("start_time"), default=9 * 60)
        end_min = self._parse_time(payload.get("end_time"), default=18 * 60)
        constraints = payload.get("constraints") if isinstance(payload.get("constraints"), dict) else {}

        best = self._optimize_route(
            places=points["places"],
            origin=points.get("origin"),
            destination=points.get("destination"),
            mode=mode,
            start_min=start_min,
            end_min=end_min,
            constraints=constraints,
        )
        response_data = {
            "provider": self.provider,
            "mode": mode,
            "origin": points.get("origin", {}).get("name"),
            "destination": points.get("destination", {}).get("name"),
            "route_order": best["route_order"],
            "legs": best["legs"],
            "total_distance_m": best["total_distance_m"],
            "total_travel_time_min": best["total_travel_time_min"],
            "total_visit_time_min": best["total_visit_time_min"],
            "estimated_finish_time": self._format_time(best["finish_min"]),
            "feasible": best["feasible"],
            "dropped_places": [],
            "warnings": self._dedupe_texts([*self._provider_warnings, *best["warnings"]]),
            "reason": best["reason"],
        }
        answer = json.dumps(response_data, ensure_ascii=False, indent=2)
        return {
            "success": True,
            "answer": answer,
            "data": {
                "route": "map_route",
                "sources": [
                    {
                        "doc_id": f"map_route:{mode}:{':'.join(best['route_order'])}",
                        "chunk_id": f"map_route:{mode}:{':'.join(best['route_order'])}",
                        "title": "Map route optimization result",
                        "source_path": self.provider,
                        "file_name": "map_route",
                    "source_type": "route_plan",
                    "text": self._source_text(response_data),
                    "score": None,
                    }
                ],
                "metadata": {
                    "provider": self.provider,
                    "configured": self._provider_configured(),
                    "mode": mode,
                    "place_count": len(points["places"]),
                    "route_order": best["route_order"],
                    "feasible": best["feasible"],
                    "warning_count": len(response_data["warnings"]),
                    "query": query,
                },
            },
        }

    def _resolve_places(self, payload: dict[str, Any], query: str) -> list[dict[str, Any]]:
        raw_places = payload.get("places")
        if not raw_places:
            raw_places = payload.get("waypoints")
        if not raw_places:
            raw_places = payload.get("entities")
        places: list[dict[str, Any]] = []
        if isinstance(raw_places, list):
            for item in raw_places:
                parsed = self._parse_place(item)
                if parsed:
                    places.append(parsed)

        if not places:
            for name in self._extract_known_places(query):
                places.append({"name": name})
        previous_text = self._previous_records_text(payload.get("previous_records"))
        if len(places) < 2:
            for name in self._extract_known_places(previous_text):
                places.append({"name": name})
        if len(places) < 2:
            for name in self._default_places_for_query(f"{query}\n{previous_text}"):
                places.append({"name": name})
        return self._dedupe_places(places)

    @staticmethod
    def _previous_records_text(previous_records: Any) -> str:
        if not isinstance(previous_records, list):
            return ""
        chunks: list[str] = []
        for record in previous_records:
            if not isinstance(record, dict):
                continue
            chunks.append(str(record.get("output") or ""))
            for source in record.get("sources") or []:
                if isinstance(source, dict):
                    chunks.append(str(source.get("text") or source.get("title") or ""))
        return "\n".join(chunk for chunk in chunks if chunk)

    def _default_places_for_query(self, text: str) -> list[str]:
        for keyword, places in self.DEFAULT_ROUTE_PLACES.items():
            if keyword in text:
                return list(places)
        return []

    def _infer_endpoints(self, query: str) -> dict[str, str]:
        if not query:
            return {}
        endpoints: dict[str, str] = {}
        for name in self._extract_known_places(query):
            start = query.find(name)
            if start < 0:
                continue
            before = query[max(0, start - 8) : start]
            after = query[start + len(name) : start + len(name) + 10]
            if "origin" not in endpoints and (
                any(token in before for token in ("从", "由", "起点", "住在", "酒店在"))
                or any(token in after for token in ("出发", "开始", "起步"))
            ):
                endpoints["origin"] = name
            if "destination" not in endpoints and (
                any(token in before for token in ("最后", "终点", "回到", "返回", "结束到"))
                or any(token in after for token in ("结束", "收尾", "返程"))
            ):
                endpoints["destination"] = name
        return endpoints

    @staticmethod
    def _remove_origin_from_places(places: list[dict[str, Any]], origin: str | None) -> list[dict[str, Any]]:
        if not origin:
            return places
        return [place for place in places if place.get("name") != origin]

    def _provider_configured(self) -> bool:
        if self.provider == "amap":
            return bool(self.amap_provider and self.amap_provider.configured)
        return True

    def _parse_place(self, item: Any) -> dict[str, Any] | None:
        if isinstance(item, str):
            name = item.strip()
            return {"name": name} if name else None
        if not isinstance(item, dict):
            return None
        name = self._place_name(item.get("name") or item.get("title") or item.get("place"))
        if not name:
            return None
        place = dict(item)
        place["name"] = name
        place["priority"] = self._int_value(place.get("priority"), default=3, minimum=1, maximum=5)
        place["visit_duration_min"] = self._int_value(
            place.get("visit_duration_min") or place.get("duration_min"),
            default=self.default_visit_duration_min,
            minimum=15,
            maximum=480,
        )
        if "weather_sensitive" in place:
            place["weather_sensitive"] = bool(place.get("weather_sensitive"))
        return place

    def _resolve_points(
        self,
        origin: str | None,
        destination: str | None,
        places: list[dict[str, Any]],
    ) -> tuple[dict[str, Any], list[str]]:
        missing: list[str] = []
        points: dict[str, Any] = {"places": []}
        if origin:
            resolved = self._resolve_point({"name": origin})
            if resolved:
                points["origin"] = resolved
            else:
                missing.append(origin)
        if destination:
            resolved = self._resolve_point({"name": destination})
            if resolved:
                points["destination"] = resolved
            else:
                missing.append(destination)
        for place in places:
            resolved = self._resolve_point(place)
            if resolved:
                points["places"].append(resolved)
            else:
                missing.append(str(place.get("name") or place))
        return points, missing

    def _resolve_point(self, place: dict[str, Any]) -> dict[str, Any] | None:
        name = self._place_name(place.get("name"))
        lat = self._float_value(place.get("lat") or place.get("latitude"))
        lng = self._float_value(place.get("lng") or place.get("lon") or place.get("longitude"))
        if lat is None or lng is None:
            coords = self._lookup_local_coordinates(name)
            if coords:
                lat, lng = coords
        amap_point: dict[str, Any] | None = None
        if (lat is None or lng is None) and self.amap_provider and name:
            try:
                amap_point = self.amap_provider.geocode(name)
            except Exception as exc:
                self._provider_warnings.append(f"Amap geocode failed for {name}: {exc}")
            if amap_point:
                lat = self._float_value(amap_point.get("lat"))
                lng = self._float_value(amap_point.get("lng"))
        if not name or lat is None or lng is None:
            return None
        resolved = dict(place)
        if amap_point:
            resolved.update(
                {
                    "formatted_address": amap_point.get("formatted_address"),
                    "geocode_level": amap_point.get("geocode_level"),
                    "coordinate_provider": "amap",
                }
            )
        resolved.update(
            {
                "name": name,
                "lat": lat,
                "lng": lng,
                "priority": self._int_value(place.get("priority"), default=3, minimum=1, maximum=5),
                "visit_duration_min": self._int_value(
                    place.get("visit_duration_min"),
                    default=self.default_visit_duration_min,
                    minimum=15,
                    maximum=480,
                ),
                "open_window": self._normalize_open_window(place.get("open_window")),
                "weather_sensitive": bool(place.get("weather_sensitive", False)),
            }
        )
        return resolved

    def _optimize_route(
        self,
        *,
        places: list[dict[str, Any]],
        origin: dict[str, Any] | None,
        destination: dict[str, Any] | None,
        mode: str,
        start_min: int,
        end_min: int,
        constraints: dict[str, Any],
    ) -> dict[str, Any]:
        if len(places) <= self.max_exact_places:
            candidates = itertools.permutations(places)
        else:
            candidates = [self._greedy_order(places, origin, mode)]

        best_order = None
        best_score = float("inf")
        best_eval = None
        for order in candidates:
            evaluated = self._evaluate_order(
                list(order),
                origin=origin,
                destination=destination,
                mode=mode,
                start_min=start_min,
                end_min=end_min,
                constraints=constraints,
            )
            if evaluated["score"] < best_score:
                best_score = evaluated["score"]
                best_order = list(order)
                best_eval = evaluated

        assert best_order is not None and best_eval is not None
        return best_eval

    def _evaluate_order(
        self,
        order: list[dict[str, Any]],
        *,
        origin: dict[str, Any] | None,
        destination: dict[str, Any] | None,
        mode: str,
        start_min: int,
        end_min: int,
        constraints: dict[str, Any],
    ) -> dict[str, Any]:
        current = origin
        current_min = start_min
        legs: list[dict[str, Any]] = []
        warnings: list[str] = []
        total_distance = 0
        total_travel = 0
        total_visit = 0
        score = 0.0

        for index, place in enumerate(order):
            if current and not self._same_point(current, place):
                leg = self._leg(current, place, mode)
                legs.append(leg)
                if leg.get("warning"):
                    warnings.append(str(leg["warning"]))
                total_distance += leg["distance_m"]
                total_travel += leg["duration_min"]
                current_min += leg["duration_min"]
                score += leg["duration_min"]
                score += self._pace_penalty(leg, mode, constraints)

            open_penalty, open_warning, adjusted_min = self._open_window_penalty(place, current_min)
            if open_warning:
                warnings.append(open_warning)
            score += open_penalty
            current_min = adjusted_min

            priority = self._int_value(place.get("priority"), default=3, minimum=1, maximum=5)
            score += index * max(0, 6 - priority) * 1.5
            if place.get("weather_sensitive") and constraints.get("avoid_weather_sensitive"):
                score += 60
                warnings.append(f"{place['name']} is weather-sensitive; keep an indoor alternative.")

            visit_min = self._int_value(
                place.get("visit_duration_min"),
                default=self.default_visit_duration_min,
                minimum=15,
                maximum=480,
            )
            total_visit += visit_min
            current_min += visit_min
            current = place

        if destination and current and not self._same_point(current, destination):
            leg = self._leg(current, destination, mode)
            legs.append(leg)
            if leg.get("warning"):
                warnings.append(str(leg["warning"]))
            total_distance += leg["distance_m"]
            total_travel += leg["duration_min"]
            current_min += leg["duration_min"]
            score += leg["duration_min"]
            score += self._pace_penalty(leg, mode, constraints)

        if current_min > end_min:
            overrun = current_min - end_min
            score += overrun * 4
            warnings.append(f"Estimated finish time exceeds end_time by {overrun} minutes.")

        if constraints.get("pace") == "relaxed" and total_travel > 150:
            score += (total_travel - 150) * 1.5
            warnings.append("Total travel time is high for a relaxed itinerary.")

        route_order = []
        if origin:
            route_order.append(origin["name"])
        route_order.extend(place["name"] for place in order)
        if destination and (not route_order or route_order[-1] != destination["name"]):
            route_order.append(destination["name"])

        return {
            "route_order": route_order,
            "legs": legs,
            "total_distance_m": int(total_distance),
            "total_travel_time_min": int(total_travel),
            "total_visit_time_min": int(total_visit),
            "finish_min": current_min,
            "feasible": current_min <= end_min and not any("outside open window" in item for item in warnings),
            "warnings": self._dedupe_texts(warnings),
            "reason": self._reason(route_order, total_travel, current_min, end_min),
            "score": score,
        }

    def _greedy_order(self, places: list[dict[str, Any]], origin: dict[str, Any] | None, mode: str) -> list[dict[str, Any]]:
        remaining = list(places)
        if origin:
            current = origin
            order: list[dict[str, Any]] = []
        else:
            current = remaining.pop(0)
            order = [current]
        while remaining:
            next_place = min(remaining, key=lambda item: self._leg(current, item, mode)["duration_min"])
            remaining.remove(next_place)
            order.append(next_place)
            current = next_place
        return order

    def _leg(self, start: dict[str, Any], end: dict[str, Any], mode: str) -> dict[str, Any]:
        if self.amap_provider and self.amap_provider.configured:
            try:
                return self.amap_provider.route(start, end, mode)
            except Exception as exc:
                warning = f"Amap route failed for {start['name']} -> {end['name']}; used local fallback. Error: {exc}"
                self._provider_warnings.append(warning)
                fallback = self._local_leg(start, end, mode)
                fallback["warning"] = warning
                fallback["provider"] = "local_fallback"
                return fallback
        return self._local_leg(start, end, mode)

    def _local_leg(self, start: dict[str, Any], end: dict[str, Any], mode: str) -> dict[str, Any]:
        straight_m = self._haversine_m(start["lat"], start["lng"], end["lat"], end["lng"])
        distance_m = max(1, int(straight_m * self.road_factor))
        speed = self.MODE_SPEED_KMH.get(mode, self.MODE_SPEED_KMH["taxi"])
        duration_min = max(1, int(math.ceil((distance_m / 1000) / speed * 60)))
        return {
            "from": start["name"],
            "to": end["name"],
            "mode": mode,
            "distance_m": distance_m,
            "duration_min": duration_min,
            "provider": self.provider,
        }

    def _open_window_penalty(self, place: dict[str, Any], arrive_min: int) -> tuple[float, str | None, int]:
        window = place.get("open_window")
        if not window:
            return 0.0, None, arrive_min
        open_min, close_min = window
        if arrive_min < open_min:
            wait = open_min - arrive_min
            return wait * 0.5, f"{place['name']} requires waiting {wait} minutes before opening.", open_min
        visit_min = self._int_value(
            place.get("visit_duration_min"),
            default=self.default_visit_duration_min,
            minimum=15,
            maximum=480,
        )
        if arrive_min + visit_min > close_min:
            return 240.0 + (arrive_min + visit_min - close_min), f"{place['name']} may be outside open window.", arrive_min
        return 0.0, None, arrive_min

    @staticmethod
    def _pace_penalty(leg: dict[str, Any], mode: str, constraints: dict[str, Any]) -> float:
        penalty = 0.0
        if constraints.get("pace") == "relaxed" and leg["duration_min"] > 45:
            penalty += (leg["duration_min"] - 45) * 1.2
        if constraints.get("avoid_long_walk") and mode in {"walk", "walking"} and leg["distance_m"] > 1500:
            penalty += (leg["distance_m"] - 1500) / 50
        return penalty

    def _extract_known_places(self, text: str) -> list[str]:
        hits = []
        for name in self.LOCAL_COORDINATES:
            if name and name in text:
                hits.append(name)
        return hits

    def _lookup_local_coordinates(self, name: str | None) -> tuple[float, float] | None:
        if not name:
            return None
        if name in self.LOCAL_COORDINATES:
            return self.LOCAL_COORDINATES[name]
        for known, coords in self.LOCAL_COORDINATES.items():
            if known in name or name in known:
                return coords
        return None

    @staticmethod
    def _same_point(first: dict[str, Any], second: dict[str, Any]) -> bool:
        if first.get("name") and first.get("name") == second.get("name"):
            return True
        return abs(float(first["lat"]) - float(second["lat"])) < 0.00001 and abs(
            float(first["lng"]) - float(second["lng"])
        ) < 0.00001

    @staticmethod
    def _haversine_m(lat1: float, lng1: float, lat2: float, lng2: float) -> float:
        radius_m = 6371000.0
        phi1 = math.radians(lat1)
        phi2 = math.radians(lat2)
        d_phi = math.radians(lat2 - lat1)
        d_lambda = math.radians(lng2 - lng1)
        a = math.sin(d_phi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(d_lambda / 2) ** 2
        return radius_m * 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))

    @staticmethod
    def _normalize_mode(mode: Any) -> str:
        text = str(mode or "taxi").strip().lower()
        aliases = {
            "步行": "walk",
            "walking": "walk",
            "walk": "walk",
            "打车": "taxi",
            "出租车": "taxi",
            "taxi": "taxi",
            "驾车": "driving",
            "自驾": "driving",
            "driving": "driving",
            "drive": "driving",
            "公交": "transit",
            "公共交通": "transit",
            "transit": "transit",
            "auto": "auto",
        }
        return aliases.get(text, "taxi")

    @staticmethod
    def _normalize_open_window(value: Any) -> tuple[int, int] | None:
        if not isinstance(value, (list, tuple)) or len(value) != 2:
            return None
        start = MapRouteTool._parse_time(value[0], default=-1)
        end = MapRouteTool._parse_time(value[1], default=-1)
        if start < 0 or end < 0 or end <= start:
            return None
        return start, end

    @staticmethod
    def _parse_time(value: Any, default: int) -> int:
        text = str(value or "").strip()
        if not text:
            return default
        match = re.match(r"^(\d{1,2}):(\d{2})$", text)
        if not match:
            return default
        hour = int(match.group(1))
        minute = int(match.group(2))
        if hour > 23 or minute > 59:
            return default
        return hour * 60 + minute

    @staticmethod
    def _format_time(minutes: int) -> str:
        minutes = max(0, minutes)
        hour = (minutes // 60) % 24
        minute = minutes % 60
        return f"{hour:02d}:{minute:02d}"

    @staticmethod
    def _place_name(value: Any) -> str | None:
        text = str(value or "").strip()
        return text or None

    @staticmethod
    def _float_value(value: Any) -> float | None:
        try:
            if value is None or value == "":
                return None
            return float(value)
        except (TypeError, ValueError):
            return None

    @staticmethod
    def _int_value(value: Any, *, default: int, minimum: int, maximum: int) -> int:
        try:
            parsed = int(value)
        except (TypeError, ValueError):
            parsed = default
        return min(max(parsed, minimum), maximum)

    @staticmethod
    def _dedupe_places(places: list[dict[str, Any]]) -> list[dict[str, Any]]:
        seen = set()
        result = []
        for place in places:
            name = str(place.get("name") or "").strip()
            if not name or name in seen:
                continue
            seen.add(name)
            result.append(place)
        return result

    @staticmethod
    def _dedupe_texts(values: list[str]) -> list[str]:
        seen = set()
        result = []
        for value in values:
            text = str(value).strip()
            if not text or text in seen:
                continue
            seen.add(text)
            result.append(text)
        return result

    @staticmethod
    def _reason(route_order: list[str], total_travel: int, finish_min: int, end_min: int) -> str:
        if finish_min <= end_min:
            return (
                f"Selected this order to reduce backtracking across {len(route_order)} stops; "
                f"estimated travel time is {total_travel} minutes."
            )
        return (
            f"Selected the lowest-cost order found, but the route may exceed the available time; "
            f"estimated travel time is {total_travel} minutes."
        )

    @staticmethod
    def _source_text(response_data: dict[str, Any]) -> str:
        lines = [
            f"provider: {response_data.get('provider')}",
            f"mode: {response_data.get('mode')}",
            f"route_order: {' -> '.join(response_data.get('route_order') or [])}",
            f"total_travel_time_min: {response_data.get('total_travel_time_min')}",
            f"feasible: {response_data.get('feasible')}",
            f"reason: {response_data.get('reason')}",
        ]
        for leg in response_data.get("legs") or []:
            lines.append(
                f"{leg.get('from')} -> {leg.get('to')}: "
                f"{leg.get('duration_min')} min, {leg.get('distance_m')} m, mode={leg.get('mode')}"
            )
        warnings = response_data.get("warnings") or []
        if warnings:
            lines.append("warnings: " + "; ".join(str(item) for item in warnings))
        return "\n".join(lines)

    def _error_result(
        self,
        *,
        reason: str,
        message: str,
        query: str,
        metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        payload = {
            "error": reason,
            "message": message,
            "provider": self.provider,
        }
        return {
            "success": False,
            "answer": json.dumps(payload, ensure_ascii=False, indent=2),
            "error": message,
            "data": {
                "route": "map_route",
                "sources": [],
                "metadata": {
                    "provider": self.provider,
                    "configured": True,
                    "query": query,
                    "reason": reason,
                    **(metadata or {}),
                },
            },
        }


class _HTMLToTextParser(HTMLParser):
    block_tags = {
        "address", "article", "aside", "blockquote", "br", "div", "dl", "dt", "dd",
        "figcaption", "figure", "footer", "form", "h1", "h2", "h3", "h4", "h5", "h6",
        "header", "hr", "li", "main", "nav", "ol", "p", "pre", "section", "table",
        "tbody", "td", "tfoot", "th", "thead", "tr", "ul",
    }

    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.parts: list[str] = []
        self.title_parts: list[str] = []
        self.skip_depth = 0
        self.in_title = False
        self.link_href_stack: list[str | None] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        tag = tag.lower()
        if tag in {"script", "style", "noscript", "template", "svg"}:
            self.skip_depth += 1
            return
        if self.skip_depth:
            return
        if tag == "title":
            self.in_title = True
        if tag in self.block_tags:
            self._newline()
        if tag == "li":
            self.parts.append("- ")
        if tag == "a":
            attrs_dict = dict(attrs)
            self.link_href_stack.append(attrs_dict.get("href"))

    def handle_endtag(self, tag: str) -> None:
        tag = tag.lower()
        if tag in {"script", "style", "noscript", "template", "svg"} and self.skip_depth:
            self.skip_depth -= 1
            return
        if self.skip_depth:
            return
        if tag == "title":
            self.in_title = False
        if tag == "a" and self.link_href_stack:
            href = self.link_href_stack.pop()
            if href:
                self.parts.append(f" ({href})")
        if tag in self.block_tags:
            self._newline()

    def handle_data(self, data: str) -> None:
        if self.skip_depth:
            return
        text = re.sub(r"\s+", " ", unescape(data)).strip()
        if not text:
            return
        if self.in_title:
            self.title_parts.append(text)
            return
        if self.parts and not self.parts[-1].endswith((" ", "\n", "- ")):
            self.parts.append(" ")
        self.parts.append(text)

    def _newline(self) -> None:
        if self.parts and not self.parts[-1].endswith("\n"):
            self.parts.append("\n")

    @property
    def title(self) -> str | None:
        title = " ".join(self.title_parts).strip()
        return title or None

    def text(self) -> str:
        raw = "".join(self.parts)
        lines = [re.sub(r"[ \t]+", " ", line).strip() for line in raw.splitlines()]
        compact_lines = []
        blank_seen = False
        for line in lines:
            if not line:
                if not blank_seen and compact_lines:
                    compact_lines.append("")
                blank_seen = True
                continue
            compact_lines.append(line)
            blank_seen = False
        return "\n".join(compact_lines).strip()


class WebFetchTool:
    def __init__(self):
        self.name = "web_fetch"
        self.timeout_seconds = float(os.getenv("WEB_FETCH_TIMEOUT_SECONDS", "10"))
        self.max_bytes = int(os.getenv("WEB_FETCH_MAX_BYTES", "1048576"))
        self.max_chars = int(os.getenv("WEB_FETCH_MAX_CHARS", "12000"))
        self.max_redirects = int(os.getenv("WEB_FETCH_MAX_REDIRECTS", "3"))
        self.user_agent = os.getenv(
            "WEB_FETCH_USER_AGENT",
            "local-rag-web-fetch/1.0",
        )

    def structured_search(self, payload: dict[str, Any]) -> dict[str, Any]:
        url, resolve_error = self._resolve_url(payload)
        if resolve_error:
            return self._error_result(url, resolve_error["reason"], resolve_error["message"])
        if not url:
            return self._error_result(None, "missing_url", "web_fetch 需要 url 或 source_index。")

        validation_error = self._validate_url(url)
        if validation_error:
            return self._error_result(url, "unsafe_url", validation_error)

        try:
            response_data = self._fetch(url)
            content = self._convert_content(response_data)
            text, truncated = self._truncate(content["text"])
        except Exception as exc:
            return self._error_result(url, "fetch_failed", str(exc))

        result = {
            "url": response_data["final_url"],
            "requested_url": url,
            "status_code": response_data["status_code"],
            "content_type": response_data["content_type"],
            "title": content.get("title"),
            "text": text,
            "truncated": truncated,
            "bytes_read": response_data["bytes_read"],
            "charset": response_data["charset"],
        }
        answer = json.dumps(result, ensure_ascii=False, indent=2)
        source_text = text[:1000] if text else None
        return {
            "success": True,
            "answer": answer,
            "data": {
                "route": "web_fetch",
                "sources": [
                    {
                        "doc_id": f"web_fetch:{response_data['final_url']}",
                        "chunk_id": f"web_fetch:{response_data['final_url']}",
                        "title": content.get("title") or response_data["final_url"],
                        "url": response_data["final_url"],
                        "source_path": response_data["final_url"],
                        "file_name": content.get("title") or response_data["final_url"],
                        "source_type": "web",
                        "text": source_text,
                        "score": None,
                    }
                ],
                "metadata": {
                    "provider": "httpx",
                    "configured": True,
                    "requested_url": url,
                    "final_url": response_data["final_url"],
                    "status_code": response_data["status_code"],
                    "content_type": response_data["content_type"],
                    "bytes_read": response_data["bytes_read"],
                    "max_bytes": self.max_bytes,
                    "max_chars": self.max_chars,
                    "truncated": truncated,
                },
            },
        }

    def _resolve_url(self, payload: dict[str, Any]) -> tuple[str | None, dict[str, str] | None]:
        url = self._extract_url(payload)
        if url:
            return url, None

        if "source_index" not in payload:
            return None, None

        try:
            source_index = int(payload.get("source_index"))
        except (TypeError, ValueError):
            return None, {
                "reason": "invalid_source_index",
                "message": "source_index 必须是从 1 开始的正整数。",
            }
        if source_index < 1:
            return None, {
                "reason": "invalid_source_index",
                "message": "source_index 必须是从 1 开始的正整数。",
            }

        url_sources = self._url_sources_from_previous_records(payload.get("previous_records"))
        if not url_sources:
            return None, {
                "reason": "missing_previous_sources",
                "message": "source_index 需要前序任务提供带 URL 的 sources。",
            }
        if source_index > len(url_sources):
            return None, {
                "reason": "source_index_out_of_range",
                "message": f"source_index={source_index} 越界，当前只有 {len(url_sources)} 条可用 URL source。",
            }
        return url_sources[source_index - 1], None

    def _extract_url(self, payload: dict[str, Any]) -> str | None:
        for key in ("url", "link"):
            value = payload.get(key)
            if isinstance(value, str) and value.strip():
                return value.strip()
        text = str(payload.get("description") or payload.get("query") or "")
        match = re.search(r"https?://[^\s<>'\")]+", text)
        if match:
            return match.group(0).rstrip(".,;，。；")
        return None

    def _url_sources_from_previous_records(self, previous_records: Any) -> list[str]:
        if not isinstance(previous_records, list):
            return []
        urls: list[str] = []
        for record in previous_records:
            if not isinstance(record, dict):
                continue
            for source in record.get("sources") or []:
                if not isinstance(source, dict):
                    continue
                url = source.get("url") or source.get("source_path")
                if isinstance(url, str) and url.startswith(("http://", "https://")):
                    urls.append(url)
        return urls

    def _validate_url(self, url: str) -> str | None:
        parsed = urlparse(url)
        if parsed.scheme not in {"http", "https"}:
            return "只允许 http 或 https URL。"
        if not parsed.hostname:
            return "URL 缺少 hostname。"
        host = parsed.hostname.strip().lower()
        if host in {"localhost", "localhost.localdomain"} or host.endswith(".localhost"):
            return "不允许访问 localhost。"
        try:
            ip = ipaddress.ip_address(host)
        except ValueError:
            return None
        if (
            ip.is_private
            or ip.is_loopback
            or ip.is_link_local
            or ip.is_multicast
            or ip.is_reserved
            or ip.is_unspecified
        ):
            return "不允许访问内网、本机或保留地址。"
        return None

    def _fetch(self, url: str) -> dict[str, Any]:
        current_url = url
        headers = {
            "user-agent": self.user_agent,
            "accept": "text/html,application/json,text/plain,text/markdown,*/*;q=0.8",
        }
        for redirect_index in range(self.max_redirects + 1):
            with httpx.stream(
                "GET",
                current_url,
                headers=headers,
                timeout=self.timeout_seconds,
                follow_redirects=False,
            ) as response:
                status_code = response.status_code
                if status_code in {301, 302, 303, 307, 308}:
                    if redirect_index >= self.max_redirects:
                        raise ValueError("redirect limit exceeded")
                    location = response.headers.get("location")
                    if not location:
                        raise ValueError("redirect response missing Location header")
                    next_url = urljoin(current_url, location)
                    validation_error = self._validate_url(next_url)
                    if validation_error:
                        raise ValueError(f"unsafe redirect URL: {validation_error}")
                    current_url = next_url
                    continue

                response.raise_for_status()
                content_type = response.headers.get("content-type", "")
                content_length = response.headers.get("content-length")
                if content_length and int(content_length) > self.max_bytes:
                    raise ValueError(f"response too large: content-length={content_length}")

                chunks: list[bytes] = []
                bytes_read = 0
                for chunk in response.iter_bytes():
                    if not chunk:
                        continue
                    bytes_read += len(chunk)
                    if bytes_read > self.max_bytes:
                        raise ValueError(f"response too large: exceeded {self.max_bytes} bytes")
                    chunks.append(chunk)

                return {
                    "final_url": str(response.url),
                    "status_code": status_code,
                    "content_type": content_type,
                    "body": b"".join(chunks),
                    "bytes_read": bytes_read,
                    "charset": response.encoding or "utf-8",
                }
        raise ValueError("redirect handling failed")

    def _convert_content(self, response_data: dict[str, Any]) -> dict[str, Any]:
        content_type = str(response_data["content_type"]).lower()
        charset = response_data.get("charset") or "utf-8"
        body = response_data["body"]
        text = body.decode(charset, errors="replace")

        if "text/html" in content_type or "application/xhtml+xml" in content_type:
            parser = _HTMLToTextParser()
            parser.feed(text)
            parser.close()
            return {"title": parser.title, "text": parser.text()}

        if "application/json" in content_type or content_type.endswith("+json"):
            try:
                parsed = json.loads(text)
                normalized = json.dumps(parsed, ensure_ascii=False, indent=2)
            except Exception:
                normalized = text
            return {"title": None, "text": normalized}

        if content_type.startswith("text/") or "xml" in content_type:
            return {"title": None, "text": text.strip()}

        raise ValueError(f"unsupported content type: {response_data['content_type'] or 'unknown'}")

    def _truncate(self, text: str) -> tuple[str, bool]:
        if len(text) <= self.max_chars:
            return text, False
        return text[: self.max_chars].rstrip() + "\n\n[内容已截断]", True

    def _error_result(self, url: str | None, reason: str, message: str) -> dict[str, Any]:
        return {
            "success": False,
            "answer": json.dumps(
                {
                    "url": url,
                    "error": reason,
                    "message": message,
                },
                ensure_ascii=False,
                indent=2,
            ),
            "error": message,
            "data": {
                "route": "web_fetch",
                "sources": [],
                "metadata": {
                    "provider": "httpx",
                    "configured": True,
                    "requested_url": url,
                    "reason": reason,
                },
            },
        }


TOOL_SPECS = {
    "local_search": ToolSpec(
        name="local_search",
        description="在本地知识图谱和文档中检索特定实体的详细信息和局部关系。",
        use_when=[
            "查询具体景点、票价、交通、开放时间、儿童票、退改规则等稳定事实",
            "需要本地知识库和图谱证据支撑的微观问题",
        ],
        avoid_when=[
            "需要宏观总结或多景点整体推荐时优先考虑 global_search",
            "需要今天、明天、最新公告等实时信息时优先考虑 web_search",
        ],
        parameters={
            "description": "检索任务描述",
            "entities": "相关实体名称列表",
        },
        plan_policy=PlanPolicy(
            importance="mergeable",
            merge_strategy="same_tool_merge_entities",
            max_instances=2,
            dedupe_keys=("task_type", "entities"),
            group="retrieval",
            drop_priority=40,
        ),
    ),
    "global_search": ToolSpec(
        name="global_search",
        description="在本地知识图谱中检索整体概念、社区摘要和跨景点主题信息。",
        use_when=[
            "查询城市级推荐、路线框架、多景点比较、整体概况",
            "需要宏观社区摘要支撑的规划或总结问题",
        ],
        avoid_when=[
            "查询某个具体票价、电话、码头等细节事实时优先考虑 local_search",
            "需要实时公告或最新开放状态时优先考虑 web_search",
        ],
        parameters={
            "description": "宏观检索任务描述",
            "entities": "主题或区域实体列表",
        },
        plan_policy=PlanPolicy(
            importance="mergeable",
            merge_strategy="same_tool_merge_query",
            max_instances=1,
            dedupe_keys=("task_type",),
            group="retrieval",
            drop_priority=30,
        ),
    ),
    "web_search": ToolSpec(
        name="web_search",
        description="查询本地知识库之外的实时网页信息，并返回带 URL 的网页来源。",
        use_when=[
            "用户明确询问最新、今天、明天、现在、是否开放",
            "查询临时闭园、天气影响、节假日公告、最新票价、最新演出场次、交通调整",
        ],
        avoid_when=[
            "普通景点介绍、稳定票务规则、历史资料等本地知识库可回答的问题",
            "没有强时效要求的路线规划或景点推荐",
        ],
        parameters={
            "description": "面向网页搜索的查询描述",
            "entities": "需要查询的景点、项目或地点",
        },
        plan_policy=PlanPolicy(
            importance="required",
            merge_strategy="chain_group",
            max_instances=1,
            dedupe_keys=("query",),
            group="web_evidence",
            requires_followup=("source_select", "web_fetch"),
            drop_priority=15,
            realtime_sensitive=True,
        ),
    ),
    "weather_query": ToolSpec(
        name="weather_query",
        description="Query weather forecast for a city or scenic area over a specific travel date range and return route-planning hints.",
        use_when=[
            "User asks for a travel itinerary, route plan, or multi-day schedule with concrete dates or relative dates such as today, tomorrow, weekend, or next week",
            "The plan should be adjusted for rain, heat, wind, storms, or other weather-sensitive outdoor activities",
        ],
        avoid_when=[
            "User only asks static attraction facts, ticket rules, or historical/cultural background without travel dates",
            "The travel date is far beyond the available forecast range; return the limitation instead of fabricating weather",
        ],
        parameters={
            "location": "City, district, or scenic area name; if omitted, infer from the user query or entities",
            "start_date": "Forecast start date in YYYY-MM-DD; relative dates may be inferred from the query",
            "end_date": "Forecast end date in YYYY-MM-DD; if omitted, infer from trip duration or use start_date",
            "query": "Original user query, used for date/location inference",
        },
        plan_policy=PlanPolicy(
            importance="required",
            merge_strategy="none",
            max_instances=1,
            dedupe_keys=("location", "start_date", "end_date"),
            group="weather",
            drop_priority=10,
            realtime_sensitive=True,
        ),
    ),
    "map_route": ToolSpec(
        name="map_route",
        description="Optimize the visit order for multiple attractions and estimate route legs, distance, travel time, feasibility, and route warnings.",
        use_when=[
            "User asks how to order multiple attractions, hotels, stations, or scenic spots in one itinerary",
            "The plan needs route feasibility, backtracking reduction, distance, travel time, or start/end point constraints",
            "A detailed itinerary needs evidence for why one attraction order is better than another",
        ],
        avoid_when=[
            "User only asks for a single attraction introduction, ticket rule, or historical background",
            "There are no candidate places to order or route between",
        ],
        parameters={
            "origin": "Optional start point, such as a hotel, station, airport, or first scenic spot",
            "destination": "Optional end point; use return_to_origin=true for a loop route",
            "places": "List of candidate places; each item may include name, visit_duration_min, priority, open_window, lat/lng",
            "mode": "walk, taxi, driving, transit, or auto",
            "start_time": "Itinerary start time in HH:MM",
            "end_time": "Itinerary end time in HH:MM",
            "constraints": "Optional pace, avoid_long_walk, avoid_weather_sensitive, with_elderly, or other route constraints",
        },
        plan_policy=PlanPolicy(
            importance="required",
            merge_strategy="none",
            max_instances=1,
            dedupe_keys=("origin", "destination", "places", "mode"),
            group="route_optimization",
            requires_previous=("local_search", "global_search"),
            requires_followup=("reflection",),
            drop_priority=12,
            realtime_sensitive=True,
        ),
    ),
    "source_select": ToolSpec(
        name="source_select",
        description="基于用户问题和前序 URL sources，选择最适合后续抓取的一条网页来源。",
        use_when=[
            "web_search 或其他工具返回多个 URL sources，需要决定后续 web_fetch 读取哪一条",
            "用户强调官方、权威、文档、公告或来源可靠性时，用 source_select 选择最合适来源",
        ],
        avoid_when=[
            "用户已经直接提供明确 URL 时不需要 source_select",
            "前序任务没有返回 URL sources 时不要使用 source_select",
        ],
        parameters={
            "description": "来源选择任务描述",
            "previous_records": "前序执行记录，由执行器自动传入",
        },
        plan_policy=PlanPolicy(
            importance="required",
            merge_strategy="none",
            max_instances=1,
            group="web_evidence",
            requires_previous=("web_search",),
            requires_followup=("web_fetch",),
            drop_priority=15,
            realtime_sensitive=True,
        ),
    ),
    "web_fetch": ToolSpec(
        name="web_fetch",
        description="读取一个明确指定的网页来源正文，将 HTML/JSON/Text 转成适合 LLM 阅读的 JSON 文本。",
        use_when=[
            "web_search 或其他前序工具已经返回 URL sources，需要进一步读取其中某一条页面正文",
            "用户直接提供 URL，并要求总结、核对或抽取页面内容",
        ],
        avoid_when=[
            "没有 url 且没有 source_index 时不要使用 web_fetch",
            "普通本地知识库可回答的问题不需要抓取网页",
            "不要抓取内网、本机、非 http/https 或未知二进制资源",
        ],
        parameters={
            "url": "需要抓取的 http/https URL；用户直接给 URL 时使用",
            "source_index": "从前序 execution records 的 URL sources 中读取第 N 条，使用从 1 开始的序号",
            "description": "也可以从任务描述中提取显式 URL；没有显式 URL 时必须提供 source_index",
        },
        plan_policy=PlanPolicy(
            importance="required",
            merge_strategy="none",
            max_instances=1,
            dedupe_keys=("url", "source_index"),
            group="web_evidence",
            requires_previous=("web_search", "source_select"),
            drop_priority=15,
            realtime_sensitive=True,
        ),
    ),
    "reflection": ToolSpec(
        name="reflection",
        description="基于已完成子任务结果做综合、冲突检查和信息缺口提示，不引入新资料。",
        use_when=[
            "多个检索任务完成后，需要综合交通、票价、时间和路线",
            "需要检查已有结果是否冲突或是否缺少关键信息",
        ],
        avoid_when=[
            "需要新增本地知识检索时使用 local_search 或 global_search",
            "需要实时网页信息时使用 web_search",
        ],
        parameters={
            "description": "反思或综合任务描述",
            "previous_records": "已有子任务执行记录，由执行器自动传入",
        },
        plan_policy=PlanPolicy(
            importance="critical",
            merge_strategy="none",
            max_instances=1,
            group="synthesis",
            drop_priority=0,
        ),
    ),
}


TOOL_REGISTRY = {
    "local_search": lambda: ToolAdapter("hybrid", "local_search"),
    "global_search": lambda: ToolAdapter("global", "global_search"),
    "web_search": lambda: WebSearchTool(),
    "weather_query": lambda: WeatherQueryTool(),
    "map_route": lambda: MapRouteTool(),
    "source_select": lambda: SourceSelectTool(),
    "web_fetch": lambda: WebFetchTool(),
    "reflection": lambda: ReflectionTool(),
}

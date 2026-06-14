import ast
import json

import jieba.posseg as pseg

from agent_system.integration.llm_factory import get_text_llm
from core.config import neo4j_password, neo4j_url, neo4j_user
from core.neo4j_handler import Neo4jHandler
from .source_utils import dedupe_sources, source_from_chunk_record
from .types import QueryResult


LOCAL_PROMPT = """
你是一个文旅问答助手。请结合知识图谱上下文和原文来源回答问题。
优先使用结构化图谱关系；如果信息不足，请明确说明。

知识图谱上下文：
{graph_context}

原文片段：
{source_context}

用户问题：
{question}
"""


class GraphLocalQueryEngine:
    def __init__(self, llm=None, neo4j_handler=None):
        self.llm = llm or get_text_llm()
        self.neo4j = neo4j_handler or Neo4jHandler(neo4j_url, neo4j_user, neo4j_password)

    def extract_entities(self, query: str) -> list[str]:
        prompt = f"""
请从用户问题中抽取文旅领域核心实体，如景点、地点、票种、码头、交通工具、公司或政策名称。
只返回 JSON 数组字符串，不要解释。

用户问题：{query}
"""
        try:
            response = self.llm.invoke(prompt)
            start = response.find("[")
            end = response.rfind("]") + 1
            if start >= 0 and end > start:
                raw = response[start:end]
                try:
                    values = json.loads(raw)
                except json.JSONDecodeError:
                    values = ast.literal_eval(raw)
                return [str(item).strip() for item in values if str(item).strip()]
        except Exception:
            pass

        words = pseg.cut(query)
        return [word.word for word in words if word.flag.startswith("n") and len(word.word) > 1]

    def resolve_entities(self, entities: list[str]) -> list[str]:
        resolved = []
        for entity in entities:
            match = self.neo4j.fuzzy_search(entity)
            if match and match not in resolved:
                resolved.append(match)
        return resolved

    def _fetch_source_chunks(self, chunk_ids: list[str], limit: int = 5):
        if not chunk_ids:
            return []
        with self.neo4j.driver.session() as session:
            result = session.run(
                """
                MATCH (c:Chunk)<-[:HAS_CHUNK]-(d:Document)
                WHERE c.id IN $chunk_ids
                RETURN c.id AS chunk_id,
                       c.doc_id AS doc_id,
                       c.chunk_index AS chunk_index,
                       c.text AS text,
                       c.source_path AS source_path,
                       c.metadata_json AS metadata_json,
                       d.name AS doc_name
                ORDER BY c.chunk_index
                LIMIT $limit
                """,
                chunk_ids=chunk_ids,
                limit=limit,
            )
            return [record.data() for record in result]

    def search(self, query: str, limit: int = 20) -> QueryResult:
        extracted = self.extract_entities(query)
        entities = self.resolve_entities(extracted)

        if not entities:
            return QueryResult(
                answer="当前问题没有命中知识图谱中的明确实体，建议改用普通向量检索。",
                route="graph_local",
                metadata={"extracted_entities": extracted, "resolved_entities": []},
            )

        graph_context = self.neo4j.get_local_context(entities, limit=limit)
        chunk_ids = []
        with self.neo4j.driver.session() as session:
            result = session.run(
                """
                MATCH (s:Entity)-[r]-(t:Entity)
                WHERE s.name IN $names
                UNWIND coalesce(r.source_ref_ids, []) AS chunk_id
                RETURN DISTINCT chunk_id
                LIMIT 10
                """,
                names=entities,
            )
            chunk_ids = [record["chunk_id"] for record in result if record["chunk_id"]]

        chunk_records = self._fetch_source_chunks(chunk_ids)
        sources = dedupe_sources([source_from_chunk_record(record) for record in chunk_records])
        source_context = "\n\n".join(
            f"[{index + 1}] {source.text}"
            for index, source in enumerate(sources)
            if source.text
        )
        answer = self.llm.invoke(
            LOCAL_PROMPT.format(
                graph_context=graph_context,
                source_context=source_context,
                question=query,
            )
        )
        return QueryResult(
            answer=answer,
            route="graph_local",
            sources=sources,
            contexts=[graph_context, source_context],
            metadata={"extracted_entities": extracted, "resolved_entities": entities},
        )

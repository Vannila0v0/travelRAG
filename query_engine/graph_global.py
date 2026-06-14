import jieba

from agent_system.integration.llm_factory import get_text_llm
from core.config import neo4j_password, neo4j_url, neo4j_user
from core.neo4j_handler import Neo4jHandler
from .source_utils import dedupe_sources, source_from_chunk_record
from .types import QueryResult, Source


GLOBAL_PROMPT = """
你是一个文旅规划与分析助手。请基于“社区摘要”和“原始支撑片段”回答用户的宏观问题。

要求：
1. 优先使用社区摘要形成整体结构。
2. 使用原始支撑片段补充具体项目、票务、交通、演出或游船信息。
3. 如果原始支撑片段没有覆盖某个细节，请明确说明资料不足，不要编造。

社区摘要：
{community_context}

原始支撑片段：
{source_context}

用户问题：{question}
"""


QUERY_EXPANSIONS = {
    "夜游": ["晚上", "夜间", "演出", "游船", "灯光", "千古情", "两江四湖"],
    "夜间": ["夜游", "晚上", "演出", "游船", "灯光", "千古情", "两江四湖"],
    "一日游": ["路线", "行程", "上午", "下午", "晚上", "交通", "票价"],
    "路线": ["行程", "交通", "票价", "景点", "安排"],
    "推荐": ["适合", "项目", "景点", "线路"],
}


class GraphGlobalQueryEngine:
    def __init__(self, llm=None, neo4j_handler=None):
        self.llm = llm or get_text_llm()
        self.neo4j = neo4j_handler or Neo4jHandler(neo4j_url, neo4j_user, neo4j_password)

    def _tokens(self, query: str) -> list[str]:
        tokens: list[str] = []
        for token in jieba.lcut(query):
            token = token.strip()
            if len(token) > 1 and token not in tokens:
                tokens.append(token)

        for trigger, expanded in QUERY_EXPANSIONS.items():
            if trigger in query:
                for token in expanded:
                    if token not in tokens:
                        tokens.append(token)
        return tokens

    def retrieve_communities(self, query: str, limit: int = 8):
        tokens = self._tokens(query)
        is_night_query = "夜游" in query or "夜间" in query or "晚上" in query
        is_city_route_query = (
            ("一日游" in query or "路线" in query or "行程" in query or "怎么玩" in query)
            and ("桂林" in query or "市区" in query)
        )
        with self.neo4j.driver.session() as session:
            result = session.run(
                """
                MATCH (c:Community)
                WHERE c.summary IS NOT NULL
                OPTIONAL MATCH (c)-[:HAS_MEMBER]->(e:Entity)
                WITH c, collect(DISTINCT coalesce(e.name, "") + " " + coalesce(e.description, "")) AS entity_texts
                WITH c,
                     reduce(text = "", item IN entity_texts | text + "\n" + item) AS entity_text,
                     $tokens AS tokens
                WITH c,
                     [token IN tokens
                         WHERE coalesce(c.title, "") CONTAINS token
                            OR coalesce(c.summary, "") CONTAINS token
                            OR coalesce(c.full_content, "") CONTAINS token
                            OR entity_text CONTAINS token] AS hits,
                     [token IN tokens
                         WHERE coalesce(c.title, "") CONTAINS token
                            OR coalesce(c.summary, "") CONTAINS token] AS strong_hits
                WITH c,
                     size(strong_hits) * 3
                     + size(hits)
                     + CASE WHEN coalesce(c.title, "") CONTAINS $query_text THEN 8 ELSE 0 END
                     + CASE WHEN coalesce(c.summary, "") CONTAINS $query_text THEN 5 ELSE 0 END AS score
                WITH c,
                     score
                     + CASE
                         WHEN $is_night_query
                              AND (
                                  coalesce(c.title, "") CONTAINS "千古情"
                                  OR coalesce(c.summary, "") CONTAINS "千古情"
                                  OR coalesce(c.full_content, "") CONTAINS "千古情"
                              )
                         THEN 12 ELSE 0 END
                     + CASE
                         WHEN $is_night_query
                              AND (
                                  coalesce(c.title, "") CONTAINS "两江四湖"
                                  OR coalesce(c.summary, "") CONTAINS "两江四湖"
                                  OR coalesce(c.full_content, "") CONTAINS "两江四湖"
                              )
                         THEN 8 ELSE 0 END AS score
                WITH c,
                     score
                     + CASE
                         WHEN $is_city_route_query
                              AND (
                                  coalesce(c.title, "") CONTAINS "桂林市区"
                                  OR coalesce(c.title, "") CONTAINS "水域管理"
                                  OR coalesce(c.summary, "") CONTAINS "桂林市区"
                              )
                         THEN 20 ELSE 0 END
                     + CASE
                         WHEN $is_city_route_query
                              AND (
                                  coalesce(c.title, "") CONTAINS "两江四湖"
                                  OR coalesce(c.summary, "") CONTAINS "两江四湖"
                              )
                         THEN 14 ELSE 0 END
                     - CASE
                         WHEN $is_city_route_query
                              AND (
                                  coalesce(c.title, "") CONTAINS "阳朔"
                                  OR coalesce(c.title, "") CONTAINS "龙胜"
                                  OR coalesce(c.title, "") CONTAINS "遇龙河"
                              )
                         THEN 8 ELSE 0 END AS score
                WHERE score > 0
                RETURN c.id AS id,
                       c.title AS title,
                       c.summary AS summary,
                       c.full_content AS full_content,
                       c.level AS level,
                       score
                ORDER BY score DESC, c.level DESC, c.id ASC
                LIMIT $limit
                """,
                query_text=query,
                tokens=tokens,
                is_night_query=is_night_query,
                is_city_route_query=is_city_route_query,
                limit=limit,
            )
            return [record.data() for record in result]

    def retrieve_supporting_chunks(
        self,
        community_ids: list,
        query: str,
        chunks_per_community: int = 4,
    ):
        if not community_ids:
            return []

        tokens = self._tokens(query)
        with self.neo4j.driver.session() as session:
            result = session.run(
                """
                UNWIND $community_ids AS cid
                MATCH (c:Community {id: cid})-[:HAS_MEMBER]->(e:Entity)-[:MENTIONED_IN]->(chunk:Chunk)<-[:HAS_CHUNK]-(d:Document)
                WITH cid, chunk, d, collect(DISTINCT e.name) AS entity_names, $tokens AS tokens
                WITH cid, chunk, d, entity_names,
                     size([token IN tokens
                           WHERE coalesce(chunk.text, "") CONTAINS token
                              OR coalesce(d.name, "") CONTAINS token
                              OR any(name IN entity_names WHERE name CONTAINS token)]) AS token_hits,
                     size(entity_names) AS mention_count
                WITH cid, collect({
                    chunk_id: chunk.id,
                    doc_id: chunk.doc_id,
                    chunk_index: chunk.chunk_index,
                    text: chunk.text,
                    source_path: chunk.source_path,
                    metadata_json: chunk.metadata_json,
                    doc_name: d.name,
                    token_hits: token_hits,
                    mention_count: mention_count,
                    score: token_hits * 10 + mention_count
                }) AS rows
                UNWIND rows AS row
                WITH cid, row
                ORDER BY cid, row.score DESC, row.chunk_index ASC
                WITH cid, collect(row)[0..$chunks_per_community] AS top_rows
                UNWIND top_rows AS row
                RETURN row.chunk_id AS chunk_id,
                       row.doc_id AS doc_id,
                       row.chunk_index AS chunk_index,
                       row.text AS text,
                       row.source_path AS source_path,
                       row.metadata_json AS metadata_json,
                       row.doc_name AS doc_name,
                       row.score AS score,
                       cid AS community_id
                ORDER BY row.score DESC, row.chunk_index ASC
                """,
                community_ids=community_ids,
                tokens=tokens,
                chunks_per_community=chunks_per_community,
            )
            return [record.data() for record in result]

    def search(self, query: str, limit: int = 8) -> QueryResult:
        communities = self.retrieve_communities(query, limit=limit)
        community_ids = [item["id"] for item in communities]
        chunk_records = self.retrieve_supporting_chunks(community_ids, query)

        community_context = "\n\n".join(
            f"[Community {item['id']}] {item.get('title')}\n{item.get('summary')}"
            for item in communities
        )
        chunk_sources = dedupe_sources([source_from_chunk_record(record) for record in chunk_records])
        source_context = "\n\n".join(
            f"[Source {index + 1}] {source.file_name}#{source.chunk_index}\n{source.text}"
            for index, source in enumerate(chunk_sources)
            if source.text
        )

        answer = self.llm.invoke(
            GLOBAL_PROMPT.format(
                community_context=community_context,
                source_context=source_context,
                question=query,
            )
        )

        community_sources = [
            Source(
                doc_id=f"community-{item['id']}",
                chunk_id=f"community-{item['id']}",
                file_name=item.get("title"),
                section="community_summary",
                text=item.get("summary"),
                score=float(item.get("score") or 0),
            )
            for item in communities
        ]
        sources = dedupe_sources(community_sources + chunk_sources)
        return QueryResult(
            answer=answer,
            route="graph_global",
            sources=sources,
            contexts=[community_context, source_context],
            metadata={
                "community_count": len(communities),
                "supporting_chunk_count": len(chunk_sources),
                "community_ids": community_ids,
            },
        )

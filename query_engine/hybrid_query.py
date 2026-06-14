from agent_system.integration.llm_factory import get_text_llm
from .graph_local import GraphLocalQueryEngine
from .source_utils import dedupe_sources
from .types import QueryResult
from .vector_query import VectorQueryEngine


HYBRID_PROMPT = """
你是一个文旅问答助手。请综合普通文档检索和知识图谱检索结果回答问题。
若两者冲突，请优先使用带明确实体关系和来源的图谱信息；若资料不足，请说明。

向量检索上下文：
{vector_context}

图谱检索上下文：
{graph_context}

用户问题：
{question}
"""


class HybridQueryEngine:
    def __init__(self, vector_engine: VectorQueryEngine, local_engine: GraphLocalQueryEngine, llm=None):
        self.vector_engine = vector_engine
        self.local_engine = local_engine
        self.llm = llm or get_text_llm()

    def search(self, query: str, vector_k: int = 4) -> QueryResult:
        docs, vector_sources = self.vector_engine.retrieve(query, k=vector_k)
        local_result = self.local_engine.search(query)
        vector_context = "\n\n".join(doc.page_content for doc in docs)
        graph_context = "\n\n".join(local_result.contexts)
        answer = self.llm.invoke(
            HYBRID_PROMPT.format(
                vector_context=vector_context,
                graph_context=graph_context,
                question=query,
            )
        )
        return QueryResult(
            answer=answer,
            route="hybrid",
            sources=dedupe_sources(vector_sources + local_result.sources),
            contexts=[vector_context, graph_context],
            metadata={"local": local_result.metadata},
        )

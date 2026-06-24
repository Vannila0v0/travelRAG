from pathlib import Path
import re
from threading import Lock

from langchain_community.vectorstores import FAISS

from agent_system.integration.llm_factory import get_embeddings_model, get_text_llm
from .source_utils import dedupe_sources, source_from_document
from .types import QueryResult, Source


DEFAULT_PROMPT = """
你是一个文旅问答助手。请严格依据给定参考资料回答用户问题。
如果资料中没有答案，请说明当前资料不足，不要编造。

参考资料：
{context}

用户问题：
{question}
"""


class VectorQueryEngine:
    def __init__(self, index_dir: str | Path = ".cache/faiss_index", llm=None):
        self.index_dir = Path(index_dir)
        self.llm = llm or get_text_llm()
        self._vectorstore = None
        self._vectorstore_lock = Lock()

    @property
    def vectorstore(self):
        if self._vectorstore is None:
            with self._vectorstore_lock:
                if self._vectorstore is None:
                    if not self.index_dir.exists():
                        raise FileNotFoundError(f"FAISS index not found: {self.index_dir}")
                    self._vectorstore = FAISS.load_local(
                        str(self.index_dir),
                        get_embeddings_model(),
                        allow_dangerous_deserialization=True,
                    )
        return self._vectorstore

    def _lexical_boost(self, query: str, text: str) -> float:
        query = query.lower()
        text = text.lower()
        boost = 0.0
        terms: set[str] = set()
        entity_terms: set[str] = set()

        for segment in re.findall(r"[\u4e00-\u9fff]{2,}", query):
            entity_terms.update(re.findall(r"[\u4e00-\u9fff]{1,8}(?:温泉|景区|四湖|千古情|银子岩|天门山)", segment))
            max_len = min(6, len(segment))
            for size in range(2, max_len + 1):
                for start in range(0, len(segment) - size + 1):
                    terms.add(segment[start:start + size])

        for token in re.findall(r"[a-z0-9]{2,}", query):
            terms.add(token)

        for term in terms:
            if term in text:
                boost += min(len(term), 6) / 10

        for entity in entity_terms:
            if entity in text:
                boost += 4.0
                continue
            suffix = next((ending for ending in ("温泉", "景区", "四湖", "千古情", "银子岩", "天门山") if entity.endswith(ending)), None)
            if suffix and re.search(rf"[\u4e00-\u9fff]{{1,8}}{suffix}", text):
                boost -= 2.0

        return min(boost, 8.0)

    def _rerank(self, query: str, results: list[tuple[object, float | None]]) -> list[tuple[object, float | None]]:
        def adjusted(item):
            doc, score = item
            base_score = float(score) if score is not None else 0.0
            text = getattr(doc, "page_content", "") or ""
            return base_score - (self._lexical_boost(query, text) * 0.04)

        return sorted(results, key=adjusted)

    def retrieve(self, query: str, k: int = 5) -> tuple[list, list[Source]]:
        candidate_k = max(k, min(k * 4, 30))
        results = self.vectorstore.similarity_search_with_score(query, k=candidate_k)
        results = self._rerank(query, results)[:k]
        docs = [doc for doc, _ in results]
        sources = [
            source_from_document(doc, score=float(score) if score is not None else None)
            for doc, score in results
        ]
        return docs, dedupe_sources(sources)

    def search(self, query: str, k: int = 5) -> QueryResult:
        docs, sources = self.retrieve(query, k=k)
        context = "\n\n".join(
            f"[{index + 1}] {doc.page_content}"
            for index, doc in enumerate(docs)
        )
        answer = self.llm.invoke(DEFAULT_PROMPT.format(context=context, question=query))
        return QueryResult(
            answer=answer,
            route="vector",
            sources=sources,
            contexts=[doc.page_content for doc in docs],
            metadata={"k": k},
        )

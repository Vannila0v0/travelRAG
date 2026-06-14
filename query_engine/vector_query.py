from pathlib import Path
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

    def retrieve(self, query: str, k: int = 5) -> tuple[list, list[Source]]:
        results = self.vectorstore.similarity_search_with_score(query, k=k)
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

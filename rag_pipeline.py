# rag_pipeline.py

from query_rewrite import rewrite_query
from multi_query import generate_multi_queries
from retriever import retrieve_with_multi_query
from llm import get_llm

def rag_answer(question: str, retriever):
    # 1. Query Rewrite
    rewritten = rewrite_query(question)

    # 2. Multi-Query Expansion
    queries = generate_multi_queries(rewritten)

    # 3. Retrieval
    docs = retrieve_with_multi_query(retriever, queries)

    context = "\n\n".join(doc.page_content for doc in docs)

    prompt = f"""
你是一个严格基于上下文回答问题的助手。
如果上下文中没有答案，请明确说明“不知道”。

上下文：
{context}

问题：
{question}
"""

    llm = get_llm()
    return llm(prompt)

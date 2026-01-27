import os
import sys

# 导入你的核心组件
from core.neo4j_handler import neo4j_client  # 使用你现有的单例
from query_rag_docling_neo4j import llm, hybrid_retrieve, rerank  # 复用你旧文件里初始化好的资源
from graph_engine.local_search import LocalSearch


def main():
    print("=== 初始化 GraphRAG Local Search 引擎 ===")

    # 1. 定义向量检索适配器
    # LocalSearch 类需要一个简单的函数接口，我们把复杂的 hybrid_retrieve 封装一下
    def vector_adapter(query):
        # 复用你 query_rag_docling_neo4j.py 里写好的混合检索
        recall_docs = hybrid_retrieve(query)
        # 复用 Rerank
        final_docs = rerank(query, recall_docs)
        return final_docs

    # 2. 实例化搜索引擎
    # 这里把 LLM, Neo4j, 和上面的向量检索函数传进去
    search_engine = LocalSearch(
        llm=llm,
        neo4j_handler=neo4j_client,
        vector_search_func=vector_adapter
    )

    print("✅ 引擎启动成功！")

    # 3. 交互循环
    while True:
        q = input("\n请输入问题 (exit退出): ")
        if q.lower() in ["exit", "quit"]:
            break

        try:
            answer = search_engine.search(q)
            print("\n🤖 AI 回答:")
            print(answer)
        except Exception as e:
            print(f"❌ 发生错误: {e}")


if __name__ == "__main__":
    main()
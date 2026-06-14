import os
import sys
import json

# 导入组件
from core.neo4j_handler import neo4j_client
from query_rag_docling_neo4j import llm, hybrid_retrieve, rerank
from graph_engine.local_search import LocalSearch
from graph_engine.global_search import GlobalSearch  # [新增]


# [新增] 路由函数
def route_query(query):
    """
    判断是具体问题(Local)还是宏观问题(Global)
    """
    prompt = f"""
    请判断用户问题的类型。

    - "LOCAL": 具体的事实查询，如“门票多少钱”、“去哪里坐车”、“象鼻山在哪”。
    - "GLOBAL": 宏观的总结、概括、比较，如“这个景区有什么特色”、“游客主要投诉什么”、“全面介绍一下票务政策”。

    用户问题: {query}

    只输出 "LOCAL" 或 "GLOBAL"，不要其他废话。
    """
    try:
        res = llm.invoke(prompt).strip().upper()
        if "GLOBAL" in res: return "GLOBAL"
        return "LOCAL"
    except:
        return "LOCAL"  # 默认兜底


    # 1. 向量适配器
def vector_adapter(query):
    #混合索引，结合了BM25和余弦相似度
    recall_docs = hybrid_retrieve(query)
    #BGE-rerank重拍
    final_docs = rerank(query, recall_docs)
    return final_docs

def main():
    print("=== 初始化 GraphRAG 引擎 (Local + Global) ===")


    # 2. 实例化两个引擎
    local_engine = LocalSearch(llm, neo4j_client, vector_adapter)
    global_engine = GlobalSearch(llm, neo4j_client)  # [新增]

    print("✅ 引擎启动成功！")

    while True:
        q = input("\n请输入问题 (exit退出): ")
        if q.lower() in ["exit", "quit"]: break

        try:
            # 3. 路由选择
            mode = route_query(q)
            print(f"🔄 路由判断: [{mode} SEARCH]")

            if mode == "GLOBAL":
                answer = global_engine.search(q)
            else:
                answer = local_engine.search(q)

            print("\n🤖 AI 回答:")
            print(answer)

        except Exception as e:
            print(f"❌ 发生错误: {e}")
            import traceback
            traceback.print_exc()


if __name__ == "__main__":
    main()
import sys
import os
import time

# 路径 Hack，确保能导入项目根目录的模块
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.neo4j_handler import Neo4jHandler
from core.config import neo4j_password, neo4j_url, neo4j_user
# 只导入检索函数，不导入原来的 llm
from query_rag_docling_neo4j import hybrid_retrieve
from agent_system.integration.llm_factory import get_llm_model

# ================= 配置区域 =================
NEO4J_URI = neo4j_url
NEO4J_USER = neo4j_user
NEO4J_PASSWORD = neo4j_password
# ===========================================

def get_enrichment_llm():
    """初始化描述补全模型。"""
    return get_llm_model()


def enrich_missing_descriptions():
    print("=== 开始执行图谱描述补全 (Enrichment - Powered by DeepSeek) ===")

    # 1. 初始化资源
    handler = Neo4jHandler(NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD)
    llm = get_enrichment_llm()

    # 2. 找出需要补全的节点
    print("1. 正在扫描缺失描述的节点...")
    cypher_find = """
    MATCH (n:Entity)
    WHERE n.description IS NULL 
       OR n.description = "" 
       OR n.description STARTS WITH "这是一个" 
    RETURN n.name AS name, labels(n) AS labels, elementId(n) AS id
    """

    nodes_to_enrich = []
    with handler.driver.session() as session:
        result = session.run(cypher_find)
        for record in result:
            # 排除 Entity 标签，找具体类型
            labels = [l for l in record['labels'] if l != 'Entity']
            type_label = labels[0] if labels else "未知类型"

            nodes_to_enrich.append({
                "name": record['name'],
                "type": type_label,
                "id": record['id']
            })

    print(f"   -> 发现 {len(nodes_to_enrich)} 个节点需要补全。")

    if not nodes_to_enrich:
        print("   -> 没有需要补全的节点，程序退出。")
        return

    # 3. 循环处理
    print("2. 开始利用 RAG + DeepSeek 生成描述...")

    for i, node in enumerate(nodes_to_enrich):
        name = node['name']
        print(f"   [{i + 1}/{len(nodes_to_enrich)}] 处理: {name} ...", end="", flush=True)

        try:
            # A. 检索上下文 (Retrieve)
            # 复用你本地的混合检索 (Faiss + BM25)
            # 注意：这里需要 ensure query_rag_docling_neo4j 里的 vectorstore 已经初始化
            docs = hybrid_retrieve(name, vec_k=3, bm25_k=3, rrf_k=60)

            # 如果没找到文档，跳过
            if not docs:
                print(" ❌ 未找到相关文档，跳过")
                continue

            # 拼接上下文，限制长度防止 Kimi 超 token
            context_text = "\n".join([d.page_content[:300] for d in docs[:3]])

            # B. 生成描述 (Generate via DeepSeek)
            prompt = f"""
            你是一个知识图谱构建助手。请根据以下参考文档，为实体 "{name}" 写一段简短的描述。

            参考文档：
            {context_text}

            要求：
            1. 描述要客观、准确，完全基于文档内容。
            2. 长度控制在 50-100 字以内。
            3. 如果文档里没提到该实体，请返回 "暂无详细信息"。
            4. 不要包含 "根据文档"、"文档提到" 等废话，直接输出描述内容。

            描述：
            """

            description = llm.invoke(prompt).content.strip()

            # 简单的清洗
            description = description.replace("\n", " ").replace('"', "'")

            if "暂无详细信息" in description or len(description) < 5:
                print(" ⚠️ 信息不足 (DeepSeek 认为文档不相关)")
                continue

            # C. 更新数据库 (Update)
            with handler.driver.session() as session:
                session.run("""
                MATCH (n)
                WHERE elementId(n) = $id
                SET n.description = $desc,
                    n.is_enriched = true
                """, id=node['id'], desc=description)

            print(" ✅ 已更新")
            # 稍微 sleep 一下防止并发过快（Kimi 默认限制较宽，但保险起见）
            time.sleep(0.5)

        except Exception as e:
            print(f" ❌ 出错: {e}")

    print("=== 补全完成！请重新运行社区发现和摘要生成 ===")
    handler.close()


if __name__ == "__main__":
    enrich_missing_descriptions()

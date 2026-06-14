import sys
import os
import time

# 路径 Hack，确保能导入 core
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.neo4j_handler import Neo4jHandler
from core.config import neo4j_password, neo4j_url, neo4j_user
from llm import get_llm

# 配置 (确保与你的环境一致)
NEO4J_URI = neo4j_url
NEO4J_USER = neo4j_user
NEO4J_PASSWORD = neo4j_password
LLM_MODEL = "deepseek-chat"

# === Prompt 定义: 让 LLM 总结社区 ===
COMMUNITY_SUMMARIZE_PROMPT = """
你是一个专业的数据分析师。以下是某个知识图谱“社区”内的实体和关系列表。
这个社区代表了某些紧密相关的概念（如特定的票务规则、某个地理区域的景点、或某种交通方式）。

请根据提供的信息，生成一份**社区摘要报告**。

### 社区数据:
{context}

### 要求:
1. **生成标题**: 给这个社区起一个简短的、概括性的标题（例如：“两江四湖夜游票务体系”或“桂林市区交通枢纽”）。
2. **生成摘要**: 详细总结这个社区包含的主要信息、规则或特征。摘要应当涵盖社区内最重要的实体和关系。
3. **格式**: 请严格按照以下 JSON 格式返回，不要包含 Markdown 标记。

{{
    "title": "社区标题",
    "summary": "这里是详细的社区摘要内容..."
}}

请直接返回 JSON 字符串。
"""


def generate_community_summaries():
    print("=== Step 5: Community Summarization (生成社区摘要) ===")

    # 1. 初始化
    handler = Neo4jHandler(NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD)
    llm = get_llm()

    # 2. 获取所有社区 ID
    print("1. 获取社区列表...")
    community_ids = []
    with handler.driver.session() as session:
        # 查出所有 id，按 id 排序
        result = session.run("MATCH (c:Community) RETURN c.id as id ORDER BY c.id")
        community_ids = [record['id'] for record in result]

    print(f"   -> 共有 {len(community_ids)} 个社区需要生成摘要。")

    if len(community_ids) == 0:
        print("   ❌ 未找到社区，请先运行 Step 4 (run_community.py)。")
        return

    # 3. 循环处理每个社区
    print("2. 开始生成摘要 (这可能需要一些时间)...")

    success_count = 0

    for cid in community_ids:
        try:
            # A. 提取社区上下文 (Context)
            # 逻辑：查找属于该社区的实体，以及这些实体之间的内部关系
            cypher_context = """
                MATCH (c:Community {id: $cid})-[:HAS_MEMBER]->(e:Entity)
                WITH collect(e) as nodes
                UNWIND nodes as s
                MATCH (s)-[r]->(t:Entity)
                WHERE t IN nodes
                RETURN s.name, s.description, type(r), r.description, t.name
                LIMIT 50 
            """
            # limit 50 是为了防止特大社区撑爆 LLM 窗口，生产环境可以用更复杂的切分策略

            context_text = []
            with handler.driver.session() as session:
                res = session.run(cypher_context, cid=cid)
                for record in res:
                    # 拼凑成文本： "夜游成人票(描述...) --[价格]--> 210元"
                    s_desc = f"({record['s.description']})" if record['s.description'] else ""
                    r_desc = f"({record['r.description']})" if record['r.description'] else ""
                    line = f"{record['s.name']}{s_desc} --[{record['type(r)']}{r_desc}]--> {record['t.name']}"
                    context_text.append(line)

            # 如果社区太小（没关系，只有孤点），就只把实体名字列出来
            if not context_text:
                with handler.driver.session() as session:
                    res = session.run(
                        "MATCH (c:Community {id: $cid})-[:HAS_MEMBER]->(e) RETURN e.name, e.description LIMIT 20",
                        cid=cid)
                    for record in res:
                        desc = f"({record['e.description']})" if record['e.description'] else ""
                        context_text.append(f"实体: {record['e.name']} {desc}")

            full_context = "\n".join(context_text)

            # B. 调用 LLM 生成摘要
            response = llm.invoke(COMMUNITY_SUMMARIZE_PROMPT.format(context=full_context))

            # C. 解析 JSON (简单清洗)
            import json
            clean_json = response.replace("```json", "").replace("```", "").strip()
            # 简单的截断修复（防止 LLM 废话前缀）
            if clean_json.find("{") != -1:
                clean_json = clean_json[clean_json.find("{"): clean_json.rfind("}") + 1]

            data = json.loads(clean_json)
            title = data.get("title", f"社区 {cid}")
            summary = data.get("summary", "无摘要")

            # D. 写回 Neo4j
            with handler.driver.session() as session:
                session.run("""
                    MATCH (c:Community {id: $cid})
                    SET c.title = $title, 
                        c.summary = $summary,
                        c.full_content = $context 
                """, cid=cid, title=title, summary=summary, context=full_context)

            print(f"   [✓] 社区 {cid}: {title}")
            success_count += 1

        except Exception as e:
            print(f"   [x] 社区 {cid} 处理失败: {e}")
            continue

    print(f"✅ Step 5 完成！成功生成 {success_count}/{len(community_ids)} 个社区摘要。")
    handler.close()


if __name__ == "__main__":
    generate_community_summaries()

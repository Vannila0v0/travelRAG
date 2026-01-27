import sys
import os
import json
import logging
from langchain_openai import ChatOpenAI  # 使用 Kimi 或 OpenAI
from langchain_core.prompts import PromptTemplate

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from core.neo4j_handler import Neo4jHandler

# ================= 配置 =================
NEO4J_URI = "bolt://localhost:7687"
NEO4J_USER = "neo4j"
NEO4J_PASSWORD = "password123"
KIMI_API_KEY = "sk-9Qoie8kvG68ou8wwr2jhgZsuJTmC7tBWAAyFimjhntQmL07x"  # 填入你的 Key

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger(__name__)

# === Prompt 1: 底层摘要 (针对实体和关系) ===
LEVEL_0_PROMPT = """
你是一个专业的数据分析师。请根据以下【实体与关系】生成一份社区摘要。

### 社区数据:
{context}

### 要求:
1. 标题：简短概括（如“两江四湖票务规则”）。
2. 摘要：详细总结社区内的核心信息、价格、规则、地点关系。
3. 格式：严格返回 JSON。
{{ "title": "...", "summary": "..." }}
"""

# === Prompt 2: 高层摘要 (针对下级摘要) ===
# 这就是递归的核心：它的输入是别人的摘要！
LEVEL_UP_PROMPT = """
你是一个宏观战略分析师。你正在阅读一份由【下级社区摘要】组成的报告。
请综合这些子社区的信息，生成一份更高层级的全局概览。

### 下级社区汇报:
{context}

### 要求:
1. 标题：高度概括的宏观标题（如“桂林旅游综合服务体系”）。
2. 摘要：不要罗列细节，而是提取共性、趋势和宏观结构。例如“该区域涵盖了从票务到交通的完整闭环...”。
3. 格式：严格返回 JSON。
{{ "title": "...", "summary": "..." }}
"""


def get_llm():
    return ChatOpenAI(
        model="moonshot-v1-8k",
        openai_api_key=KIMI_API_KEY,
        openai_api_base="https://api.moonshot.cn/v1",
        temperature=0.1
    )


def recursive_summarize():
    logger.info("=== Step 5: 递归社区摘要 (Recursive Summarization) ===")

    handler = Neo4jHandler(NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD)
    llm = get_llm()

    # 1. 确定有多少层级
    levels = []
    with handler.driver.session() as session:
        res = session.run("MATCH (c:Community) RETURN DISTINCT c.level as lvl ORDER BY lvl")
        levels = [r['lvl'] for r in res]

    logger.info(f"检测到层级: {levels} (将采用 Bottom-Up 策略)")

    # 2. 自底向上处理 (Bottom-Up)
    for lvl in levels:
        logger.info(f"--- 正在处理 Level {lvl} ---")

        # 获取该层级所有社区 ID
        cids = []
        with handler.driver.session() as session:
            res = session.run("MATCH (c:Community) WHERE c.level = $lvl RETURN c.id as id", lvl=lvl)
            cids = [r['id'] for r in res]

        for cid in cids:
            try:
                # ------------------------------------------------
                # 分支逻辑：底层用实体，高层用子摘要
                # ------------------------------------------------
                context_text = ""

                if lvl == 0:
                    # === Level 0: 读实体和边 ===
                    cypher = """
                        MATCH (c:Community {id: $cid})-[:HAS_MEMBER]->(e:Entity)
                        OPTIONAL MATCH (e)-[r]->(target:Entity)
                        WHERE (c)-[:HAS_MEMBER]->(target) // 仅限社区内部关系
                        RETURN e.name, e.description, type(r), target.name
                        LIMIT 100
                    """
                    lines = []
                    with handler.driver.session() as session:
                        res = session.run(cypher, cid=cid)
                        for r in res:
                            rel_str = f"--[{r['type(r)']}]--> {r['target.name']}" if r['target.name'] else ""
                            lines.append(f"实体: {r['e.name']} ({r['e.description']}) {rel_str}")
                    context_text = "\n".join(lines)
                    prompt_template = LEVEL_0_PROMPT

                else:
                    # === Level > 0: 读下级社区摘要 (Recursive!) ===
                    # 查找通过 IN_COMMUNITY 连上来的子社区
                    cypher = """
                        MATCH (child:Community)-[:IN_COMMUNITY]->(parent:Community {id: $cid})
                        RETURN child.title, child.summary
                    """
                    lines = []
                    with handler.driver.session() as session:
                        res = session.run(cypher, cid=cid)
                        for r in res:
                            lines.append(f"### 子板块: {r['child.title']}\n内容: {r['child.summary']}")

                    # 如果没有显式 IN_COMMUNITY 关系（可能是孤立社区），降级回落实体
                    if not lines:
                        logger.warning(f"   社区 {cid} 没有检测到子社区关系，尝试使用成员实体回退...")
                        # ... (此处省略回退逻辑，简单起见跳过)
                        continue

                    context_text = "\n\n".join(lines)
                    prompt_template = LEVEL_UP_PROMPT

                if not context_text.strip():
                    logger.warning(f"   社区 {cid} 上下文为空，跳过")
                    continue

                # 3. 调用 LLM
                prompt = PromptTemplate.from_template(prompt_template)
                chain = prompt | llm
                response = chain.invoke({"context": context_text}).content

                # 4. 解析与存储
                clean_json = response.replace("```json", "").replace("```", "").strip()
                if clean_json.find("{") != -1:  # 简单提取 json
                    clean_json = clean_json[clean_json.find("{"): clean_json.rfind("}") + 1]

                data = json.loads(clean_json)

                with handler.driver.session() as session:
                    session.run("""
                        MATCH (c:Community {id: $cid})
                        SET c.title = $title, 
                            c.summary = $summary,
                            c.full_content = $ctx
                    """, cid=cid, title=data.get("title"), summary=data.get("summary"), ctx=context_text)

                logger.info(f"   [✓] L{lvl} {cid}: {data.get('title')}")

            except Exception as e:
                logger.error(f"   [x] 处理 {cid} 失败: {e}")

    logger.info("✅ 递归摘要生成完成！")
    handler.close()


if __name__ == "__main__":
    recursive_summarize()
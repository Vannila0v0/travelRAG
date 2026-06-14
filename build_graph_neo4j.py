import json
from core.neo4j_handler import neo4j_client
from llm import get_llm
# 引入之前切好的 chunks，如果报错，请确保 query_rag_docling.py 里把 chunks 暴露出来了
# 或者你可以简单的把切分逻辑复制过来
from query_rag_docling import chunks

# 1. 设置 LLM
llm = get_llm()

# 2. 定义抽取 Prompt
EXTRACT_PROMPT = """
你是一个知识图谱构建专家。请从下面的文本中提取关键实体（地点、景点、美食、活动）及其关系。

要求：
1. 输出格式必须是严格的 JSON 列表：[["实体1", "关系", "实体2"], ["实体A", "关系", "实体B"]]
2. 实体要具体（如"外滩"，不要用"该地"）
3. 关系要简洁（如：位于、包含、特色是、毗邻）
4. 只输出 JSON，不要任何解释，不要 Markdown 标记

文本：
{text}
"""


def clean_json_string(s):
    """清洗 LLM 输出，去除 markdown 符号"""
    s = s.strip()
    if s.startswith("```json"): s = s[7:]
    if s.startswith("```"): s = s[3:]
    if s.endswith("```"): s = s[:-3]
    return s.strip()


def build_graph():
    print(f"开始构建图谱，共 {len(chunks)} 个片段待处理...")

    count = 0
    for i, chunk in enumerate(chunks):
        content = chunk.page_content
        # 跳过太短的片段
        if len(content) < 50: continue

        prompt = EXTRACT_PROMPT.format(text=content)

        try:
            # 调用大模型
            response = llm.invoke(prompt)
            cleaned_res = clean_json_string(response)

            # 解析 JSON
            triples = json.loads(cleaned_res)

            # 写入 Neo4j
            for src, rel, tgt in triples:
                neo4j_client.add_triple(src, rel, tgt)
                count += 1

            print(f"[进度 {i + 1}/{len(chunks)}] 提取并写入了 {len(triples)} 条关系")

        except json.JSONDecodeError:
            print(f"[跳过] Chunk {i} LLM 输出格式错误")
        except Exception as e:
            print(f"[错误] Chunk {i}: {e}")

    print(f"构建完成！共写入 {count} 条关系。")
    print("正在执行自动化 DQA (数据质量治理)...")
    neo4j_client.perform_dqa()
    neo4j_client.close()


if __name__ == "__main__":
    build_graph()

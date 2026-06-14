# multi_query.py

from llm import get_llm
from config import MULTI_QUERY_NUM

MULTI_QUERY_PROMPT = """
你是一个检索查询生成器。
请基于下面的问题，生成 {n} 个不同表达方式但语义一致的检索查询。

要求：
1. 每个查询侧重点略有不同
2. 使用适合技术文档检索的表达
3. 不引入新事实
4. 使用换行分隔输出

问题：
{question}
"""

def generate_multi_queries(question: str) -> list[str]:
    llm = get_llm()
    prompt = MULTI_QUERY_PROMPT.format(
        question=question,
        n=MULTI_QUERY_NUM
    )

    result = llm(prompt)
    queries = [q.strip("- ").strip() for q in result.split("\n") if q.strip()]
    return queries

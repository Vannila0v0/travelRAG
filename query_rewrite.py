# query_rewrite.py

from llm import *

REWRITE_PROMPT = """
你是一个搜索查询优化助手。
你的任务是将用户问题改写为更适合在技术文档中检索的形式。

要求：
1. 保留原始语义
2. 使用偏技术、书面化表达
3. 不要添加文档中可能不存在的内容
4. 只输出重写后的问题

原始问题：
{question}
"""

def rewrite_query(question: str) -> str:
    llm = get_llm()
    prompt = REWRITE_PROMPT.format(question=question)
    return llm(prompt).strip()

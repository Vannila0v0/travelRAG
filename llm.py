# llm.py
from agent_system.integration.llm_factory import get_text_llm


def get_llm():
    """兼容旧 RAG 脚本的文本 LLM 实例。"""
    return get_text_llm()

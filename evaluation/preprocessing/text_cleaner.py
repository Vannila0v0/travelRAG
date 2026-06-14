import re


def clean_thinking_process(text: str) -> str:
    """
    清除回答中的 <think>...</think> 思考过程。
    由于你的 agent_system 可能会输出复杂的执行日志，这里也去掉了可能存在的思维链标签。
    """
    if not text:
        return ""
    # 匹配并移除 <think> 到 </think> 之间的所有内容
    cleaned = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL)
    return cleaned.strip()


def clean_references(text: str) -> str:
    """
    清除回答中的引用标记，如 [1], [Entity_123], [Chunk_456]
    这样算精确匹配度(EM)和F1时，不会因为引用标记干扰而掉分。
    """
    if not text:
        return ""
    # 移除常见的数字引用 [1], [1,2]
    cleaned = re.sub(r'\[\d+(?:,\s*\d+)*\]', '', text)
    # 移除实体/关系/Chunk引用 [Entity_xxx], [Rel_xxx], [Chunk_xxx]
    cleaned = re.sub(r'\[(?:Entity|Rel|Chunk)_[^\]]+\]', '', cleaned, flags=re.IGNORECASE)

    # 清理多余的空格
    cleaned = re.sub(r'\s+', ' ', cleaned)
    return cleaned.strip()
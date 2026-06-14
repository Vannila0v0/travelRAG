import re
from typing import Dict, List


def extract_references_from_answer(answer: str) -> Dict[str, List[str]]:
    """
    从系统回答中提取所有被引用的实体和关系ID。
    假设你的 agent 引用格式为 [Entity_123] 或 [Rel_456] 或 [Chunk_789]。
    如果你的 hybridagent 或 agent_system 使用了不同的引用格式，请在此处修改正则。
    """
    result = {
        "entities": [],
        "relationships": [],
        "chunks": []
    }

    if not answer:
        return result

    # 提取实体 (例如: [Entity_123] 或 [123])
    entity_matches = re.findall(r'\[Entity_([^\]]+)\]', answer, re.IGNORECASE)
    if entity_matches:
        result["entities"].extend(entity_matches)

    # 提取关系 (例如: [Rel_456])
    rel_matches = re.findall(r'\[Rel_([^\]]+)\]', answer, re.IGNORECASE)
    if rel_matches:
        result["relationships"].extend(rel_matches)

    # 提取文本块 (例如: [Chunk_789])
    chunk_matches = re.findall(r'\[Chunk_([^\]]+)\]', answer, re.IGNORECASE)
    if chunk_matches:
        result["chunks"].extend(chunk_matches)
        # 有时 Chunk 也被当作一种广义的实体来评估覆盖率
        result["entities"].extend(chunk_matches)

    # 去重
    result["entities"] = list(set(result["entities"]))
    result["relationships"] = list(set(result["relationships"]))
    result["chunks"] = list(set(result["chunks"]))

    return result
import hashlib
import json
from typing import Any, List, Optional
from .base import CacheKeyStrategy


class ContextAwareCacheStrategy(CacheKeyStrategy):
    """
    上下文感知策略：基于查询文本 + 执行上下文生成 Key。

    防止“缓存污染”：避免不同 Agent 角色、不同模型参数下的结果混用。
    """

    def __init__(self, important_keys: Optional[List[str]] = None):
        """
        Args:
            important_keys: 指定 kwargs 中哪些字段需要参与 Key 计算。
                            默认包含 ['model', 'temperature', 'agent_role']
        """
        self.important_keys = important_keys or [
            'cache_schema_version',
            'model',
            'temperature',
            'max_tokens',
            'agent_role',
            'system_instruction',
            'route',
            'task_type',
            'entities',
            'intent_tags',
            'prompt_version',
            'data_version'
        ]

    def generate_key(self, query: str, **kwargs: Any) -> str:
        # 1. 提取关键上下文
        context_data = {
            "query": query.strip(),
            "context": {}
        }

        for k in self.important_keys:
            if k in kwargs:
                val = kwargs[k]
                # 简单处理列表/字典以外的复杂对象，转为字符串
                if not isinstance(val, (str, int, float, bool, list, dict, type(None))):
                    val = str(val)
                context_data["context"][k] = val

        # 2. 序列化为确定性字符串 (sort_keys=True 是关键)
        try:
            serialized_str = json.dumps(context_data, sort_keys=True, default=str)
        except (TypeError, ValueError):
            # 降级处理：直接转字符串
            serialized_str = str(context_data)

        # 3. 生成哈希
        key_hash = hashlib.md5(serialized_str.encode('utf-8', errors='ignore')).hexdigest()

        return f"ctx_{key_hash}"

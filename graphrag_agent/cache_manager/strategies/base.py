from abc import ABC, abstractmethod
from typing import Any


class CacheKeyStrategy(ABC):
    """
    缓存键生成策略的抽象基类 (Interface)。

    不同的策略决定了如何从用户查询(query)和上下文(kwargs)中生成唯一的缓存键。
    - SimpleStrategy: 仅基于 query 的哈希。
    - ContextAwareStrategy: 结合 query + agent_id + model_name 等生成。
    - SemanticStrategy: (高级) 基于向量语义而非字符串字面量。
    """

    @abstractmethod
    def generate_key(self, query: str, **kwargs: Any) -> str:
        """
        生成唯一的缓存键。

        Args:
            query: 用户的原始查询字符串。
            **kwargs: 额外的上下文信息，如：
                     - agent_role: 智能体角色
                     - model: 使用的模型名称
                     - temperature: 温度参数

        Returns:
            str: 唯一的缓存键字符串 (通常是 hash 值)
        """
        pass

    def __repr__(self) -> str:
        return f"<{self.__class__.__name__}>"
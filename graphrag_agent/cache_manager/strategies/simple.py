import hashlib
from typing import Any
from .base import CacheKeyStrategy


class SimpleCacheStrategy(CacheKeyStrategy):
    """
    简单缓存策略：仅基于查询文本生成 Key。
    适用于：
    - 通用知识查询
    - 不依赖模型参数（如温度、角色）的确定性任务
    """

    def generate_key(self, query: str, **kwargs: Any) -> str:
        # 1. 预处理：去除首尾空格
        normalized_query = query.strip()

        # 2. 生成 MD5 哈希
        # 使用 encode('utf-8', errors='ignore') 防止特殊字符编码错误
        query_hash = hashlib.md5(normalized_query.encode('utf-8', errors='ignore')).hexdigest()

        return f"simple_{query_hash}"
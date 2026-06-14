from abc import ABC, abstractmethod
from typing import Any, Optional, Dict, Union
# 注意：这里使用相对导入，假设 CacheItem 在 ../models/cache_item.py
# 在实际运行时，Python 会处理好包路径
from ..models.cache_item import CacheItem


class CacheStorageBackend(ABC):
    """
    缓存存储后端的抽象基类 (Interface)。

    任何具体的存储实现（内存、磁盘、Redis等）都必须继承此类并实现以下方法。
    这确保了 CacheManager 可以无缝切换底层存储。
    """

    @abstractmethod
    def get(self, key: str) -> Optional[CacheItem]:
        """
        根据键获取缓存项。

        Args:
            key: 缓存键

        Returns:
            CacheItem 对象，如果未找到则返回 None
        """
        pass

    @abstractmethod
    def set(self, key: str, value: Any, metadata: Optional[Dict[str, Any]] = None) -> None:
        """
        设置缓存项。

        Args:
            key: 缓存键
            value: 缓存内容（可以是字符串、字典或 CacheItem 对象）
            metadata: 可选的元数据字典（如果 value 已经是 CacheItem，则合并）
        """
        pass

    @abstractmethod
    def delete(self, key: str) -> None:
        """
        删除指定键的缓存项。

        Args:
            key: 要删除的键
        """
        pass

    @abstractmethod
    def clear(self) -> None:
        """
        清空存储中的所有缓存数据。
        警告：这是一个破坏性操作。
        """
        pass

    @abstractmethod
    def exists(self, key: str) -> bool:
        """
        检查键是否存在于缓存中。

        Args:
            key: 缓存键

        Returns:
            bool: 存在返回 True，否则 False
        """
        pass

    def __repr__(self) -> str:
        return f"<{self.__class__.__name__}>"
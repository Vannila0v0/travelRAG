from typing import Any, Optional, Dict
from collections import OrderedDict
from .base import CacheStorageBackend
from ..models.cache_item import CacheItem


class MemoryCacheBackend(CacheStorageBackend):
    """
    基于内存的缓存后端，实现了 LRU (Least Recently Used) 淘汰策略。
    适用于高频访问、生命周期短的数据。
    """

    def __init__(self, max_items: int = 1000):
        """
        Args:
            max_items: 内存中保留的最大项目数
        """
        self._cache: OrderedDict[str, CacheItem] = OrderedDict()
        self.max_items = max_items

    def get(self, key: str) -> Optional[CacheItem]:
        if key not in self._cache:
            return None

        # LRU 关键逻辑：访问后将其移到末尾，表示"最近使用"
        self._cache.move_to_end(key)
        item = self._cache[key]
        item.update_access_stats()
        return item

    def set(self, key: str, value: Any, metadata: Optional[Dict[str, Any]] = None) -> None:
        # 如果已存在，先删除以便重新插入到末尾（或者直接 move_to_end + update）
        if key in self._cache:
            self._cache.move_to_end(key)

        # 封装为 CacheItem
        item = CacheItem.from_any(value)
        if metadata:
            item.metadata.update(metadata)

        self._cache[key] = item

        # LRU 淘汰：如果超出容量，弹出第一个（最久未使用）
        if len(self._cache) > self.max_items:
            self._cache.popitem(last=False)

    def delete(self, key: str) -> None:
        if key in self._cache:
            del self._cache[key]

    def clear(self) -> None:
        self._cache.clear()

    def exists(self, key: str) -> bool:
        return key in self._cache

    def __len__(self) -> int:
        return len(self._cache)
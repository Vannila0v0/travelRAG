from typing import Any, Optional, Dict
from .base import CacheStorageBackend
from .memory import MemoryCacheBackend
from .disk import DiskCacheBackend
from ..models.cache_item import CacheItem


class HybridCacheBackend(CacheStorageBackend):
    """
    混合缓存后端：结合内存缓存（一级，快）和磁盘缓存（二级，持久）。

    策略：
    - 读取：先查内存 -> 未命中查磁盘 -> 若磁盘命中则回填内存。
    - 写入：同时写入内存和磁盘。
    - 删除：同时从两端删除。
    """

    def __init__(self,
                 memory_backend: Optional[CacheStorageBackend] = None,
                 disk_backend: Optional[CacheStorageBackend] = None,
                 **kwargs):
        """
        Args:
            memory_backend: 可选的一级缓存实例
            disk_backend: 可选的二级缓存实例
            **kwargs: 如果未提供实例，用于初始化默认后端的参数 (如 cache_dir, max_items)
        """
        self.l1 = memory_backend or MemoryCacheBackend(
            max_items=kwargs.get('max_memory_items', 1000)
        )
        self.l2 = disk_backend or DiskCacheBackend(
            cache_dir=kwargs.get('cache_dir', './.cache')
        )

    def get(self, key: str) -> Optional[CacheItem]:
        # 1. 尝试从一级缓存（内存）读取
        item = self.l1.get(key)
        if item:
            return item

        # 2. 尝试从二级缓存（磁盘）读取
        item = self.l2.get(key)
        if item:
            # Read-Through/Read-Repair: 磁盘有但内存没有，回填到内存
            self.l1.set(key, item)
            return item

        return None

    def set(self, key: str, value: Any, metadata: Optional[Dict[str, Any]] = None) -> None:
        # 同时写入两级缓存
        # 注意：CacheItem 的封装通常在具体的 backend.set 中处理，
        # 但如果 value 已经是 CacheItem，底层 set 会自动处理。
        self.l1.set(key, value, metadata)
        self.l2.set(key, value, metadata)

    def delete(self, key: str) -> None:
        self.l1.delete(key)
        self.l2.delete(key)

    def clear(self) -> None:
        self.l1.clear()
        self.l2.clear()

    def exists(self, key: str) -> bool:
        # 只要有一层存在即可
        return self.l1.exists(key) or self.l2.exists(key)

    def __repr__(self) -> str:
        return f"<HybridCacheBackend(l1={self.l1}, l2={self.l2})>"
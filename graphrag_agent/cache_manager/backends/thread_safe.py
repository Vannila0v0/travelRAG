import threading
from typing import Any, Optional, Dict
from .base import CacheStorageBackend
from ..models.cache_item import CacheItem


class ThreadSafeCacheBackend(CacheStorageBackend):
    """
    线程安全装饰器。

    将任意非线程安全的后端（如 Memory 或 Disk）包装为线程安全版本。
    使用 RLock (可重入锁) 确保同一线程可以多次获取锁，避免死锁。
    """

    def __init__(self, backend: CacheStorageBackend):
        self._backend = backend
        self._lock = threading.RLock()

    def get(self, key: str) -> Optional[CacheItem]:
        with self._lock:
            return self._backend.get(key)

    def set(self, key: str, value: Any, metadata: Optional[Dict[str, Any]] = None) -> None:
        with self._lock:
            self._backend.set(key, value, metadata)

    def delete(self, key: str) -> None:
        with self._lock:
            self._backend.delete(key)

    def clear(self) -> None:
        with self._lock:
            self._backend.clear()

    def exists(self, key: str) -> bool:
        with self._lock:
            return self._backend.exists(key)

    def __repr__(self) -> str:
        return f"<ThreadSafe({self._backend})>"
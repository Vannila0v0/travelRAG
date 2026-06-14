import os
import json
import hashlib
import tempfile
import shutil
from pathlib import Path
from typing import Any, Optional, Dict

from .base import CacheStorageBackend
from ..models.cache_item import CacheItem


class DiskCacheBackend(CacheStorageBackend):
    """
    基于文件系统的持久化缓存后端。
    将缓存项序列化为 JSON 文件存储。
    """

    def __init__(self, cache_dir: str = "./.cache"):
        self.cache_dir = Path(cache_dir)
        self._ensure_dir()

    def _ensure_dir(self):
        """确保缓存目录存在"""
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def _get_filename(self, key: str) -> Path:
        """
        将任意 Key 转换为安全的文件路径。
        使用 MD5 哈希避免文件名过长或包含非法字符。
        """
        key_hash = hashlib.md5(key.encode('utf-8')).hexdigest()
        return self.cache_dir / f"{key_hash}.json"

    def get(self, key: str) -> Optional[CacheItem]:
        filepath = self._get_filename(key)

        if not filepath.exists():
            return None

        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                data = json.load(f)
                item = CacheItem.from_dict(data)

                # 更新访问时间并回写（可选：为了性能可跳过回写，或者只在内存层更新）
                # 这里为了简化，我们在读取时只更新内存对象状态，不强制回写磁盘
                # 因为磁盘IO很贵。真正的更新会在下次 set 时发生。
                item.update_access_stats()
                return item
        except (json.JSONDecodeError, OSError):
            # 文件损坏或读取失败，视为未命中
            return None

    def set(self, key: str, value: Any, metadata: Optional[Dict[str, Any]] = None) -> None:
        item = CacheItem.from_any(value)
        if metadata:
            item.metadata.update(metadata)

        filepath = self._get_filename(key)
        json_str = item.to_json()

        # 原子写入：先写临时文件，再重命名。防止写入中断导致文件损坏。
        try:
            # 在同一目录下创建临时文件以确保 rename 是原子操作
            fd, temp_path = tempfile.mkstemp(dir=self.cache_dir, text=True)
            with os.fdopen(fd, 'w', encoding='utf-8') as f:
                f.write(json_str)

            # 替换目标文件
            os.replace(temp_path, filepath)
        except OSError:
            # 如果写入失败，尝试清理临时文件
            if os.path.exists(temp_path):
                os.remove(temp_path)

    def delete(self, key: str) -> None:
        filepath = self._get_filename(key)
        if filepath.exists():
            try:
                os.remove(filepath)
            except OSError:
                pass  # 忽略并发删除错误

    def clear(self) -> None:
        """清空缓存目录"""
        if self.cache_dir.exists():
            shutil.rmtree(self.cache_dir)
            self._ensure_dir()

    def exists(self, key: str) -> bool:
        return self._get_filename(key).exists()
import threading
import time
import logging
from contextlib import contextmanager

logger = logging.getLogger(__name__)


class ChatConcurrentManager:
    """
    会话并发管理器

    用于管理基于 session_id 的并发锁，防止同一会话同时进行多个处理。
    """
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(ChatConcurrentManager, cls).__new__(cls)
            cls._instance._initialize()
        return cls._instance

    def _initialize(self):
        # 存储每个 session_id 对应的锁
        self._locks = {}
        # 记录锁的最后访问时间，用于清理
        self._lock_access_time = {}
        # 全局锁，用于保护 _locks 字典本身的线程安全
        self._global_lock = threading.RLock()

    def get_lock(self, session_id: str) -> threading.Lock:
        """获取或创建指定 session 的锁"""
        with self._global_lock:
            if session_id not in self._locks:
                self._locks[session_id] = threading.Lock()

            # 更新最后访问时间
            self._lock_access_time[session_id] = time.time()
            return self._locks[session_id]

    def try_acquire_lock(self, session_id: str, blocking: bool = False) -> bool:
        """
        尝试获取锁
        :param blocking: 是否阻塞等待
        :return: 是否成功获取
        """
        lock = self.get_lock(session_id)
        # acquire(blocking=False) 会立即返回结果
        acquired = lock.acquire(blocking=blocking)

        if not acquired:
            logger.warning(f"Session {session_id} is busy. Lock acquisition failed.")

        return acquired

    def release_lock(self, session_id: str):
        """释放锁"""
        with self._global_lock:
            if session_id in self._locks:
                try:
                    self._locks[session_id].release()
                except RuntimeError:
                    # 锁可能已经被释放，忽略错误
                    pass

    def cleanup_expired_locks(self, timeout: int = 3600):
        """
        清理长时间未使用的锁，防止内存泄漏
        :param timeout: 超时时间（秒），默认1小时
        """
        with self._global_lock:
            current_time = time.time()
            # 找出所有超时的 session_id
            to_remove = [
                sid for sid, last_time in self._lock_access_time.items()
                if current_time - last_time > timeout
            ]

            for sid in to_remove:
                del self._locks[sid]
                del self._lock_access_time[sid]

            if to_remove:
                logger.info(f"Cleaned up {len(to_remove)} expired locks.")


# 全局单例实例
chat_manager = ChatConcurrentManager()
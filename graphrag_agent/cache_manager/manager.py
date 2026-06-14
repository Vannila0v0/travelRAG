import logging
import threading
from typing import Any, Optional, Dict, List, Tuple

from .config import CacheConfig
from .models.cache_item import CacheItem
from .backends.base import CacheStorageBackend
from .backends.memory import MemoryCacheBackend
from .backends.disk import DiskCacheBackend
from .backends.hybrid import HybridCacheBackend
from .backends.thread_safe import ThreadSafeCacheBackend

from .strategies.base import CacheKeyStrategy
from .strategies.simple import SimpleCacheStrategy
from .strategies.context_aware import ContextAwareCacheStrategy

from .vector_similarity.embeddings import BaseEmbedding, LocalHuggingFaceEmbedding, OpenAIEmbedding
from .vector_similarity.matcher import VectorMatcher

logger = logging.getLogger(__name__)


class CacheManager:
    """
    智能体缓存系统的核心管理器。

    职责：
    1. 协调存储后端 (Memory/Disk/Hybrid)
    2. 管理键生成策略 (Simple/ContextAware)
    3. 执行向量相似度检索 (Vector Matching)
    """

    def __init__(self, config: Optional[CacheConfig] = None):
        self.config = config or CacheConfig()

        # 1. 初始化存储后端
        self.backend = self._initialize_backend()

        # 2. 初始化键生成策略
        self.simple_strategy = SimpleCacheStrategy()
        self.context_strategy = ContextAwareCacheStrategy()

        # 3. 初始化向量组件 (如果启用)
        self.embedding_model: Optional[BaseEmbedding] = None
        self.matcher: Optional[VectorMatcher] = None
        self.vector_index: Dict[str, List[float]] = {}  # 内存中的向量索引 key -> embedding

        if self.config.enable_vector_match:
            self._initialize_vector_components()

    def _initialize_backend(self) -> CacheStorageBackend:
        """根据配置构建存储后端链"""
        logger.info(f"Initializing Cache Backend: {self.config.backend_type}")

        backend_map = {
            'memory': lambda: MemoryCacheBackend(max_items=self.config.max_memory_items),
            'disk': lambda: DiskCacheBackend(cache_dir=self.config.base_dir),
            'hybrid': lambda: HybridCacheBackend(
                max_memory_items=self.config.max_memory_items,
                cache_dir=self.config.base_dir
            )
        }

        base_backend = backend_map.get(self.config.backend_type, backend_map['hybrid'])()

        # 总是包裹线程安全锁，因为 Agent 环境通常是多线程的
        return ThreadSafeCacheBackend(base_backend)

    def _initialize_vector_components(self):
        """初始化嵌入模型和匹配器"""
        try:
            if self.config.embedding_backend == 'openai':
                self.embedding_model = OpenAIEmbedding()
            else:
                self.embedding_model = LocalHuggingFaceEmbedding(
                    model_name=self.config.local_embedding_model
                )

            self.matcher = VectorMatcher(self.config)
            logger.info("Vector similarity components initialized.")
        except Exception as e:
            logger.warning(f"Failed to initialize vector components: {e}. Vector search disabled.")
            self.config.enable_vector_match = False

    def _get_strategy(self, **kwargs) -> CacheKeyStrategy:
        """根据上下文决定使用哪种 Key 策略"""
        # 如果包含特定上下文参数（如 model, role），使用上下文感知策略
        if any(k in kwargs for k in ['model', 'agent_role', 'temperature']):
            return self.context_strategy
        return self.simple_strategy

    def get(self, query: str, **kwargs) -> Optional[Any]:
        """
        获取缓存内容。

        流程：
        1. 生成 Key（精确匹配）。
        2. 如果精确命中 -> 返回。
        3. 如果未命中且开启向量搜索 -> 计算 Query 向量 -> 查找相似项 -> 返回。
        """
        if not query:
            return None

        # 1. 精确查找 (Exact Match)
        strategy = self._get_strategy(**kwargs)
        key = strategy.generate_key(query, **kwargs)

        item = self.backend.get(key)
        if item and not item.is_expired(self.config.default_ttl):
            logger.debug(f"Cache HIT (Exact): {key}")
            return item.get_content()

        # 2. 向量查找 (Semantic Match)
        if self.config.enable_vector_match and self.embedding_model and self.matcher:
            return self._get_via_vector(query)

        logger.debug(f"Cache MISS: {key}")
        return None

    def _get_via_vector(self, query: str) -> Optional[Any]:
        """执行向量相似度查找"""
        # 1. 生成查询向量
        try:
            query_vector = self.embedding_model.embed_query(query)
        except Exception as e:
            logger.error(f"Embedding generation failed: {e}")
            return None

        # 2. 收集候选集
        # 注意：这里我们使用 self.vector_index 这是一个内存优化的索引
        # 实际生产中可能需要遍历 backend 或使用专门的 Vector DB
        candidates = []
        candidate_keys = []

        # 简单的遍历内存索引（针对小规模缓存有效）
        # 如果缓存很大，这里需要优化
        keys_to_check = list(self.vector_index.keys())

        # 批量获取候选 item (这可能比较重，取决于 backend 实现)
        # 优化：我们先只拿 vector 进行纯数学计算，找出最佳 key，然后再去 backend 取 item
        best_key = None
        best_score = -1.0

        # 将 index 转换为列表以便传给 matcher
        # index_vectors: List[List[float]]
        index_keys = list(self.vector_index.keys())
        index_vectors = list(self.vector_index.values())

        if not index_vectors:
            return None

        # 使用 matcher 的批量计算能力
        similarities = self.matcher.calculate_similarity_batch(query_vector, index_vectors)

        # 找到最大值
        if len(similarities) > 0:
            import numpy as np
            best_idx = np.argmax(similarities)
            best_score = float(similarities[best_idx])

            if best_score >= self.config.similarity_threshold:
                best_key = index_keys[best_idx]

        # 3. 如果命中，从后端取回完整对象
        if best_key:
            item = self.backend.get(best_key)
            if item:
                logger.info(f"Cache HIT (Vector): Score {best_score:.4f} > {self.config.similarity_threshold}")
                return item.get_content()

        return None

    def set(self, query: str, content: Any, **kwargs) -> None:
        """
        存入缓存。
        """
        if not query or content is None:
            return

        # 1. 生成 Key
        strategy = self._get_strategy(**kwargs)
        key = strategy.generate_key(query, **kwargs)

        # 2. 准备元数据
        metadata = {
            "original_query": query,
            "strategy": strategy.__class__.__name__
        }

        # 3. 生成并存储向量 (如果启用)
        if self.config.enable_vector_match and self.embedding_model:
            try:
                vector = self.embedding_model.embed_query(query)
                metadata["embedding"] = vector
                # 更新内存索引
                self.vector_index[key] = vector
            except Exception as e:
                logger.error(f"Embedding failed during set: {e}")

        # 4. 写入后端
        self.backend.set(key, content, metadata=metadata)
        logger.debug(f"Cache SET: {key}")

    def clear(self):
        """清空缓存"""
        self.backend.clear()
        self.vector_index.clear()
        logger.info("Cache cleared.")

    def delete(self, query: str, **kwargs):
        """删除指定缓存"""
        strategy = self._get_strategy(**kwargs)
        key = strategy.generate_key(query, **kwargs)
        self.backend.delete(key)
        if key in self.vector_index:
            del self.vector_index[key]

    def update_quality_score(self, query: str, score_delta: int, **kwargs) -> int:
        """
        更新缓存项的质量分数 (RLHF 机制)。

        Args:
            query: 原始查询
            score_delta: 分数变化 (例如 +1 表示点赞, -1 表示点踩)
            **kwargs: 上下文参数 (必须与生成缓存时一致，才能生成相同的 Key)

        Returns:
            int: 更新后的最新分数。如果未找到缓存，返回 0。
        """
        # 1. 重生成 Key
        strategy = self._get_strategy(**kwargs)
        key = strategy.generate_key(query, **kwargs)

        # 2. 获取现有项
        item = self.backend.get(key)
        if not item:
            logger.warning(f"Cannot update score: Cache key {key} not found.")
            return 0

        # 3. 更新分数
        current_score = item.metadata.get("quality_score", 0)
        new_score = current_score + score_delta

        item.metadata["quality_score"] = new_score

        # 记录验证状态
        if new_score > 0:
            item.metadata["user_verified"] = True

        logger.info(f"Updating quality score for '{query[:20]}...': {current_score} -> {new_score}")

        # 4. 自动淘汰机制 (可选)
        # 如果质量太差 (例如 -5分)，直接删除，避免污染后续回答
        if new_score <= -5:
            logger.info(f"Cache item quality too low ({new_score}), evicting: {key}")
            self.delete(query, **kwargs)
            return new_score

        # 5. 写回后端
        # 注意：这里我们不需要重新 embed 向量，保留原有的即可
        self.backend.set(key, item, metadata=item.metadata)

        return new_score
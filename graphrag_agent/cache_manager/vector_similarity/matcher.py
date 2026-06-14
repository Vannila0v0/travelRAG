import numpy as np
from typing import List, Tuple, Optional, Any
import logging

# 引入我们定义的数据模型和配置
from ..models.cache_item import CacheItem
from ..config import CacheConfig

logger = logging.getLogger(__name__)


class VectorMatcher:
    """
    向量相似度匹配器 (基于 NumPy 实现)。

    虽然原项目使用了 FAISS，但在轻量级 Agent 缓存场景下，
    直接使用 NumPy 进行矩阵运算可以避免复杂的索引同步问题，
    同时保持极高的计算效率（对于 <10万 级别的缓存项）。
    """

    def __init__(self, config: CacheConfig):
        """
        Args:
            config: 缓存配置对象，包含相似度阈值等设置
        """
        self.config = config
        self.threshold = config.similarity_threshold

    def calculate_similarity_batch(self,
                                   query_vector: List[float],
                                   target_vectors: List[List[float]]) -> np.ndarray:
        """
        批量计算余弦相似度。

        Formula: (A . B) / (||A|| * ||B||)

        Args:
            query_vector: 查询向量 (dim,)
            target_vectors: 目标向量列表 (n, dim)

        Returns:
            np.ndarray: 相似度分数数组 (n,)
        """
        if not target_vectors:
            return np.array([])

        # 转换为 numpy 数组
        # A: (dim,)
        A = np.array(query_vector, dtype=np.float32)
        # B: (n, dim)
        B = np.array(target_vectors, dtype=np.float32)

        # 维度校验
        if A.shape[0] != B.shape[1]:
            logger.warning(f"Vector dimension mismatch: Query {A.shape} vs Candidates {B.shape}")
            return np.zeros(len(target_vectors))

        # 计算范数 (L2 Norm)
        norm_A = np.linalg.norm(A)
        norm_B = np.linalg.norm(B, axis=1)

        # 防止除以零
        if norm_A == 0:
            return np.zeros(len(target_vectors))

        # 避免 B 中有零向量导致除以零警告
        with np.errstate(divide='ignore', invalid='ignore'):
            # 点积
            dot_products = np.dot(B, A)
            # 余弦相似度
            similarities = dot_products / (norm_A * norm_B)

        # 将无效值(NaN/Inf)置为0
        similarities = np.nan_to_num(similarities)

        return similarities

    def find_best_match(self,
                        query_vector: List[float],
                        candidates: List[CacheItem]) -> Tuple[Optional[CacheItem], float]:
        """
        在候选列表中查找最佳匹配项。

        流程：
        1. 提取所有候选项目的向量。
        2. 批量计算相似度。
        3. 找到最大分数。
        4. 判断是否超过阈值。

        Args:
            query_vector: 当前查询的向量
            candidates: 包含 metadata['embedding'] 的缓存项列表

        Returns:
            (BestMatchItem, Score)
            如果没有超过阈值的匹配，返回 (None, MaxScore)
        """
        if not candidates or not query_vector:
            return None, 0.0

        # 1. 过滤并提取有效向量
        valid_candidates = []
        valid_vectors = []

        for item in candidates:
            # 确保 item 有 embedding 且不为空
            emb = item.metadata.get("embedding")
            if emb and isinstance(emb, list) and len(emb) > 0:
                valid_candidates.append(item)
                valid_vectors.append(emb)

        if not valid_candidates:
            return None, 0.0

        # 2. 批量计算 (性能优化核心)
        try:
            scores = self.calculate_similarity_batch(query_vector, valid_vectors)
        except Exception as e:
            logger.error(f"Error calculating similarities: {e}")
            return None, 0.0

        # 3. 找到最佳匹配
        if len(scores) == 0:
            return None, 0.0

        best_idx = np.argmax(scores)
        best_score = float(scores[best_idx])
        best_item = valid_candidates[best_idx]

        # 4. 阈值判定
        if best_score >= self.threshold:
            logger.info(f"Vector match hit! Score: {best_score:.4f} >= {self.threshold}")

            # 丰富元数据，方便后续追踪
            best_item.metadata["matched_via_vector"] = True
            best_item.metadata["similarity_score"] = best_score

            return best_item, best_score
        else:
            # logger.debug(f"No vector match. Best score: {best_score:.4f} < {self.threshold}")
            return None, best_score
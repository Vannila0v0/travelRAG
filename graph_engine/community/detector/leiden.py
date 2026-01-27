import logging
import networkx as nx
from typing import Dict
from .base import BaseCommunityDetector

logger = logging.getLogger(__name__)


class LeidenDetector(BaseCommunityDetector):
    """
    基于 Leiden 算法的社区检测器。
    """

    def __init__(self, max_cluster_size: int = 10):
        self.max_cluster_size = max_cluster_size

    def detect(self, graph: nx.Graph) -> Dict[str, int]:
        logger.info(f"正在运行 Leiden 算法 (节点数: {graph.number_of_nodes()})...")

        try:
            # === 修正点 1：改用 leiden 而不是 hierarchical_leiden ===
            # leiden 函数直接返回 {node: community_id} 字典，不需要复杂解析
            from graspologic.partition import leiden

            # graspologic 的 leiden 可能需要一些特定的参数，这里保持默认即可
            # 它会自动处理网络并返回 partition 字典
            partition = leiden(graph)

            logger.info(f"Leiden 算法计算完成，发现 {len(set(partition.values()))} 个社区。")
            return partition

        except ImportError:
            logger.warning("未检测到 graspologic 库，降级使用 python-louvain...")
            return self._fallback_louvain(graph)
        except Exception as e:
            # === 修正点 2：打印具体的错误信息，方便调试 ===
            logger.error(f"Leiden 算法执行出错: {e}，正在降级使用 Louvain...")
            return self._fallback_louvain(graph)

    def _fallback_louvain(self, graph: nx.Graph) -> Dict[str, int]:
        """降级方案：使用 python-louvain"""
        try:
            import community.community_louvain as louvain
            partition = louvain.best_partition(graph)
            logger.info(f"Louvain 降级算法计算完成，发现 {len(set(partition.values()))} 个社区。")
            return partition
        except ImportError:
            # 这里的提示会告诉你缺什么包
            raise ImportError("降级失败：请运行 `pip install python-louvain` 安装所需库！")
from abc import ABC, abstractmethod
import networkx as nx
from typing import Dict


class BaseCommunityDetector(ABC):
    """
    社区检测算法的抽象基类。
    所有的具体算法（如 Leiden, SLLPA）都必须继承此类。
    """

    @abstractmethod
    def detect(self, graph: nx.Graph) -> Dict[str, int]:
        """
        执行社区检测。

        :param graph: NetworkX 图对象 (无向图)
        :return: 字典映射 {节点名: 社区ID}
        """
        pass
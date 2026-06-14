import os
from dataclasses import dataclass, field
from typing import Optional, Literal
from pathlib import Path
from dotenv import load_dotenv

# 加载根目录的 .env 文件
load_dotenv()


@dataclass
class CacheConfig:
    """
    缓存系统配置类。

    支持多层级配置：
    1. 默认值
    2. 环境变量 (.env)
    3. 运行时覆盖 (init arguments)

    这允许为不同的 Agent 实例创建不同的缓存策略。
    """

    # --- 基础存储配置 ---

    # 缓存后端类型: 'memory' (仅内存), 'disk' (仅磁盘), 'hybrid' (混合)
    backend_type: Literal['memory', 'disk', 'hybrid'] = field(
        default_factory=lambda: os.getenv("CACHE_BACKEND_TYPE", "hybrid")
    )

    # 磁盘缓存目录 (仅当 backend_type 包含 disk 时有效)
    base_dir: str = field(
        default_factory=lambda: os.getenv("CACHE_BASE_DIR", "./.cache")
    )

    # 全局过期时间 (秒)，默认 24 小时
    default_ttl: int = field(
        default_factory=lambda: int(os.getenv("CACHE_DEFAULT_TTL", 86400))
    )

    # --- 内存/混合缓存特有配置 ---

    # 内存中最大保留的项目数 (LRU策略)
    max_memory_items: int = field(
        default_factory=lambda: int(os.getenv("CACHE_MAX_MEMORY_ITEMS", 1000))
    )

    # --- 向量相似度缓存配置 ---

    # 是否启用语义/向量匹配
    enable_vector_match: bool = field(
        default_factory=lambda: os.getenv("CACHE_ENABLE_VECTOR", "true").lower() == "true"
    )

    # 向量匹配的相似度阈值 (0.0 - 1.0)
    # 较高意味着需要更相似才算命中，较低则更宽容但可能不准确
    similarity_threshold: float = field(
        default_factory=lambda: float(os.getenv("CACHE_SIMILARITY_THRESHOLD", 0.9))
    )

    # 向量化后端: 'local' (SentenceTransformer), 'openai', 'custom' (复用 llm_factory)
    embedding_backend: str = field(
        default_factory=lambda: os.getenv("CACHE_EMBEDDING_BACKEND", "local")
    )

    # 本地模型名称 (如果 embedding_backend='local')
    local_embedding_model: str = field(
        default_factory=lambda: os.getenv("CACHE_LOCAL_MODEL", "models--sentence-transformers--paraphrase-multilingual-MiniLM-L12-v2")
    )

    def __post_init__(self):
        """初始化后的验证与处理"""
        # 确保缓存目录是绝对路径或相对于当前工作目录
        self.base_dir = str(Path(self.base_dir).resolve())

        # 验证阈值范围
        if not (0 <= self.similarity_threshold <= 1.0):
            raise ValueError(f"Similarity threshold must be between 0 and 1, got {self.similarity_threshold}")

    @classmethod
    def for_agent(cls, difficulty_level: str) -> 'CacheConfig':
        """
        工厂方法：根据任务难度生成推荐的缓存配置。

        Args:
            difficulty_level: 'simple', 'medium', 'hard'
        """
        base_config = cls()

        if difficulty_level == 'simple':
            # 简单任务：完全信任缓存，低阈值，仅使用内存以求速度
            base_config.backend_type = 'memory'
            base_config.similarity_threshold = 0.85
            base_config.default_ttl = 86400 * 7  # 7天

        elif difficulty_level == 'hard':
            # 困难任务：谨慎使用缓存，高阈值，混合存储以持久化昂贵的推理结果
            base_config.backend_type = 'hybrid'
            base_config.similarity_threshold = 0.95  # 要求极高相似度
            base_config.enable_vector_match = True  # 必须开启语义匹配

        return base_config
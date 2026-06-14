import os
from pathlib import Path

# 基础路径配置
BASE_DIR = Path(__file__).parent.parent.parent
CACHE_DIR = BASE_DIR / "cache_data"


class Settings:
    # 缓存配置
    CACHE_CONFIG = {
        "dir": CACHE_DIR,
        "memory_only": False,
        "max_memory_size": 1000,
        "max_disk_size": 10000,
        "thread_safe": True
    }

    # LLM 配置 (DeepSeek OpenAI-compatible API)
    LLM_API_BASE = os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com")
    LLM_MODEL = os.getenv("DEEPSEEK_MODEL", "deepseek-chat")


# ==========================================
# [新增] Agent 运行时的参数字典 (解决 .get 报错)
# ==========================================
AGENT_SETTINGS = {
    "default_recursion_limit": 25,     # 图工作流的最大循环次数
    "stream_flush_threshold": 10,      # 流式输出时，缓存多少个字符刷新一次
    "chunk_size": 5                    # 降级流式输出的切片大小
}


# ==========================================
# 代理系统全局设置
# ==========================================

# 默认的回答排版要求
response_type = """
- 使用清晰、结构化的中文 Markdown 格式输出。
- 重要的数据、人名、地点、价格等信息请使用 **加粗** 突出显示。
- 尽量使用无序列表或有序列表来梳理多个要点，提升阅读体验。
- 语气需专业、客观且富有亲和力。
"""

settings = Settings()

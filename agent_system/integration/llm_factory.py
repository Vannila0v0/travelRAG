import os
from dotenv import load_dotenv
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_openai import ChatOpenAI
from sentence_transformers import CrossEncoder

load_dotenv()

# ==========================================
# 全局单例缓存：防止本地模型被重复加载到内存中
# ==========================================
_CHAT_MODEL_INSTANCE = None
_STREAM_CHAT_MODEL_INSTANCE = None
_EMBEDDINGS_INSTANCE = None
_RERANKER_INSTANCE = None

# ==========================================
# LLM 与本地模型路径配置
# ==========================================
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY", "")
DEEPSEEK_BASE_URL = os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com")
DEEPSEEK_MODEL = os.getenv("DEEPSEEK_MODEL", "deepseek-chat")
LLM_TEMPERATURE = float(os.getenv("LLM_TEMPERATURE", "0"))

EMBEDDING_MODEL_PATH = os.getenv(
    "EMBEDDING_MODEL_PATH",
    "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
)
RERANKER_MODEL_PATH = os.getenv(
    "RERANKER_MODEL_PATH",
    "BAAI/bge-reranker-base",
)

# 设备配置：如果你有 NVIDIA 显卡并装了 CUDA，可以改成 "cuda"
DEVICE = os.getenv("MODEL_DEVICE", "cpu")


def _require_deepseek_api_key():
    if not DEEPSEEK_API_KEY:
        raise RuntimeError(
            "DEEPSEEK_API_KEY 未配置。请在环境变量或 .env 中设置 DEEPSEEK_API_KEY。"
        )


def get_llm_model():
    """获取常规 Chat LLM 实例 (用于思考、路由、调用工具)。"""
    global _CHAT_MODEL_INSTANCE
    if _CHAT_MODEL_INSTANCE is None:
        _require_deepseek_api_key()
        _CHAT_MODEL_INSTANCE = ChatOpenAI(
            model=DEEPSEEK_MODEL,
            api_key=DEEPSEEK_API_KEY,
            base_url=DEEPSEEK_BASE_URL,
            temperature=LLM_TEMPERATURE,
        )
    return _CHAT_MODEL_INSTANCE


def get_stream_llm_model():
    """获取流式 Chat LLM 实例 (用于给前端实现打字机效果)。"""
    global _STREAM_CHAT_MODEL_INSTANCE
    if _STREAM_CHAT_MODEL_INSTANCE is None:
        _require_deepseek_api_key()
        _STREAM_CHAT_MODEL_INSTANCE = ChatOpenAI(
            model=DEEPSEEK_MODEL,
            api_key=DEEPSEEK_API_KEY,
            base_url=DEEPSEEK_BASE_URL,
            temperature=LLM_TEMPERATURE,
            streaming=True,
        )
    return _STREAM_CHAT_MODEL_INSTANCE


class TextLLMAdapter:
    """兼容旧脚本的文本 LLM 适配器。

    旧代码大量使用 OllamaLLM，并假设 invoke(prompt) 直接返回字符串。
    ChatOpenAI 返回 AIMessage，因此这里做一层轻量适配。
    """

    def __init__(self, chat_model=None):
        self.chat_model = chat_model or get_llm_model()

    def invoke(self, prompt, **kwargs) -> str:
        response = self.chat_model.invoke(prompt, **kwargs)
        return response.content if hasattr(response, "content") else str(response)

    async def ainvoke(self, prompt, **kwargs) -> str:
        response = await self.chat_model.ainvoke(prompt, **kwargs)
        return response.content if hasattr(response, "content") else str(response)

    def __call__(self, prompt, **kwargs) -> str:
        return self.invoke(prompt, **kwargs)


def get_text_llm():
    """获取兼容旧 RAG 脚本的文本 LLM。"""
    return TextLLMAdapter(get_llm_model())


def get_embeddings_model():
    """获取 Embedding 模型 (单例模式，避免重复加载)"""
    global _EMBEDDINGS_INSTANCE
    if _EMBEDDINGS_INSTANCE is None:
        print("正在加载本地 Embedding 模型到内存...")
        _EMBEDDINGS_INSTANCE = HuggingFaceEmbeddings(
            model_name=EMBEDDING_MODEL_PATH,
            model_kwargs={'device': DEVICE},
            encode_kwargs={'normalize_embeddings': True} # 推荐开启归一化，提升余弦相似度计算效果
        )
    return _EMBEDDINGS_INSTANCE

def get_reranker_model():
    """获取 Reranker 重排模型 (单例模式，避免重复加载)"""
    global _RERANKER_INSTANCE
    if _RERANKER_INSTANCE is None:
        print("正在加载本地 Reranker 模型到内存...")
        _RERANKER_INSTANCE = CrossEncoder(
            RERANKER_MODEL_PATH,
            device=DEVICE
        )
    return _RERANKER_INSTANCE

def count_tokens(text: str) -> int:
    """简单通用的 token 计数器 (纯本地备用方案)"""
    if not text:
        return 0
    # 由于不调用外部 API，使用中文字符算1个，英文单词算1/4个的经验公式估算
    chinese = len([c for c in text if '\u4e00' <= c <= '\u9fff'])
    english = len(text) - chinese
    return chinese + english // 4

if __name__ == '__main__':
    # ==========================================
    # 本地测试逻辑：运行这个文件可以检查环境是否正常
    # ==========================================
    print("1. 测试大模型连接...")
    llm = get_llm_model()
    try:
        response = llm.invoke("你好，请只回复'大模型连接成功'。")
        print(f"大模型响应: {response.content}")
    except Exception as e:
        print(f"大模型连接失败: {e}")

    print("\n2. 测试 Embedding 模型加载...")
    embed_model = get_embeddings_model()
    vector = embed_model.embed_query("测试文本")
    print(f"Embedding 成功！向量维度: {len(vector)}")

    print("\n3. 测试 Reranker 模型加载...")
    reranker = get_reranker_model()
    scores = reranker.predict([("我爱北京天安门", "北京是中国的首都")])
    print(f"Reranker 成功！相关性打分: {scores[0]}")

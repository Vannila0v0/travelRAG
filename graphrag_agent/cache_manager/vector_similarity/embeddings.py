from abc import ABC, abstractmethod
from typing import List, Union, Optional
import logging
import os

logger = logging.getLogger(__name__)


class BaseEmbedding(ABC):
    """
    向量嵌入接口基类。
    用于将文本查询转换为向量，以便进行相似度比较。
    """

    @abstractmethod
    def embed_query(self, text: str) -> List[float]:
        """将单个查询文本转换为向量"""
        pass

    @abstractmethod
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """批量将文本文档转换为向量"""
        pass


class LocalHuggingFaceEmbedding(BaseEmbedding):
    """
    基于本地 HuggingFace (Sentence-Transformers) 的嵌入实现。
    完全离线运行，无需 API Key。
    """

    def __init__(self, model_name: str = r"E:\MyOwnProj\local-rag-lab\cache\models--sentence-transformers--paraphrase-multilingual-MiniLM-L12-v2\snapshots\86741b4e3f5cb7765a600d3a3d55a0f6a6cb443d", device: Optional[str] = None):
        """
        Args:
            model_name: 本地模型名称或路径。
                        例如: 'sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2'
            device: 运行设备 'cpu', 'cuda', 'mps'。留空则自动检测。
        """
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError:
            raise ImportError(
                "Could not import sentence_transformers. "
                "Please install it with `pip install sentence-transformers`."
            )

        self.model_name = model_name

        # 自动检测设备
        if not device:
            import torch
            if torch.cuda.is_available():
                device = "cuda"
            elif torch.backends.mps.is_available():
                device = "mps"  # 适配 Mac M系列芯片
            else:
                device = "cpu"

        logger.info(f"Loading local embedding model: {model_name} on {device}")

        # 加载模型
        # 注意：如果你已经下载了模型文件在特定目录，model_name 可以是绝对路径
        self._model = SentenceTransformer(model_name, device=device)

    def embed_query(self, text: str) -> List[float]:
        # encode 返回的是 numpy array，转换为 list 方便存储 JSON
        embedding = self._model.encode(text, convert_to_numpy=True)
        return embedding.tolist()

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        embeddings = self._model.encode(texts, convert_to_numpy=True)
        return embeddings.tolist()


class OpenAIEmbedding(BaseEmbedding):
    """
    (可选) OpenAI 嵌入实现，作为备用方案。
    需要设置 OPENAI_API_KEY。
    """

    def __init__(self, model_name: str = "text-embedding-3-small"):
        try:
            from openai import OpenAI
        except ImportError:
            raise ImportError("Please install openai with `pip install openai`")

        self.client = OpenAI()
        self.model_name = model_name

    def embed_query(self, text: str) -> List[float]:
        text = text.replace("\n", " ")
        return self.client.embeddings.create(input=[text], model=self.model_name).data[0].embedding

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        # OpenAI 建议将换行符替换为空格
        texts = [t.replace("\n", " ") for t in texts]
        data = self.client.embeddings.create(input=texts, model=self.model_name).data
        return [d.embedding for d in data]
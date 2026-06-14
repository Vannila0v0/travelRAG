# config.py
import os
from dotenv import load_dotenv

load_dotenv()

DEEPSEEK_MODEL = os.getenv("DEEPSEEK_MODEL", "deepseek-chat")
DEEPSEEK_BASE_URL = os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com")
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY", "")

EMBEDDING_MODEL_PATH = os.getenv(
    "EMBEDDING_MODEL_PATH",
    "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
)
RERANKER_MODEL_PATH = os.getenv(
    "RERANKER_MODEL_PATH",
    "BAAI/bge-reranker-base",
)

CHUNK_SIZE = 500
CHUNK_OVERLAP = 50

TOP_K = 3
MULTI_QUERY_NUM = 3

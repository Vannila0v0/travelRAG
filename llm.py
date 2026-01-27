# llm.py
from langchain_ollama import OllamaLLM


def get_llm():
    return OllamaLLM(model="qwen2.5:14b",base_url="http://172.22.224.1:11434")

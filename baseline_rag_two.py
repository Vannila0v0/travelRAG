import os
# 1. 必须在导入 HuggingFaceEmbeddings 之前设置镜像环境变量
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

from langchain_ollama import OllamaLLM
# 修复过时警告：改用 langchain_huggingface
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

# 2. 读取文档
try:
    with open("data/旅游问答.md", "r", encoding="utf-8") as f:
        text = f.read()
except FileNotFoundError:
    print("错误：请确保 data/旅游问答.md 文件存在")
    exit()

docs = [Document(page_content=text)]

# 3. 切分
splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50
)
chunks = splitter.split_documents(docs)

# 4. 向量化 (添加镜像支持后，会自动从国内镜像下载)
print("正在加载/下载向量模型，请稍候...")
embedding = HuggingFaceEmbeddings(
    model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
    # 第一次运行会自动下载，建议指定 cache_folder 方便下次直接读取
    cache_folder="./cache"
)

vectorstore = FAISS.from_documents(chunks, embedding)

# 5. 检索
retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

# 6. 本地模型
# 注意：14b 模型需要较强显卡（12G显存以上），如果依然报错，请改为 qwen2.5:7b
llm = OllamaLLM(model="qwen2.5:14b")

def ask(question: str):
    try:
        docs = retriever.invoke(question)
        context = "\n\n".join(d.page_content for d in docs)

        prompt = f"""
你是一个严格基于上下文回答问题的助手。
如果上下文中没有答案，请明确说明“不知道”。

上下文：
{context}

问题：
{question}
"""
        return llm.invoke(prompt)
    except Exception as e:
        return f"发生错误：{str(e)}"

# 测试
if __name__ == "__main__":
    print("\n回答：")
    print(ask("银子岩的开放时间和售票时间是"))
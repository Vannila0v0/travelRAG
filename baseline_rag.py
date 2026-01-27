from langchain_ollama import OllamaLLM
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

# 1. 读取文档
with open("data/旅游问答.md", "r", encoding="utf-8") as f:
    text = f.read()

docs = [Document(page_content=text)]

# 2. 切分
splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50
)
chunks = splitter.split_documents(docs)
print("chunks=",chunks)
# 3. 向量化
embedding = HuggingFaceEmbeddings(
    model_name=r"E:\MyOwnProj\local-rag-lab\cache\models--sentence-transformers--paraphrase-multilingual-MiniLM-L12-v2\snapshots\86741b4e3f5cb7765a600d3a3d55a0f6a6cb443d"
)

vectorstore = FAISS.from_documents(chunks, embedding)

# 4. 检索
retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

# 5. 本地模型
llm = OllamaLLM(model="qwen2.5:14b",base_url="http://172.22.224.1:11434")

def ask(question: str):
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

# 测试
print(ask("我是学生，我想泡温泉，有推荐吗"))

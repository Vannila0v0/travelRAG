from langchain_ollama import OllamaLLM
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from rank_bm25 import BM25Okapi
from sentence_transformers import CrossEncoder
from collections import OrderedDict
import jieba
import os

# ======================
# 1. 读取文档
# ======================
file_path = "data/桂林旅游产品常用知识(1).docx"
with open(file_path, "r", encoding="gb18030") as f:
    text = f.read()

docs = [Document(page_content=text)]

# ======================
# 2. 文档切分（baseline，不动）
# ======================
splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50
)
chunks = splitter.split_documents(docs)
print(f"切分完成，共 {len(chunks)} 个片段")

# ======================
# BM25 准备工作
# ======================
# 所有 chunk 文本
bm25_corpus = [doc.page_content for doc in chunks]

# 简单分词（中文使用 jieba）
bm25_tokenized = [list(jieba.cut(text)) for text in bm25_corpus]
bm25 = BM25Okapi(bm25_tokenized)


# 用 BM25 检索 Top-N
# def bm25_retrieve(query: str, k: int = 5):
#     tokenized_query = list(jieba.cut(query))
#     scores = bm25.get_scores(tokenized_query)
#
#     ranked = sorted(
#         zip(scores, chunks),
#         key=lambda x: x[0],
#         reverse=True
#     )
#
#     # 注意：这里只返回 Document 对象列表，不含分数
#     return [doc for score, doc in ranked[:k]]

#这段返回的就是分数
def bm25_retrieve(query: str, k: int = 10) -> list[tuple[float, Document]]:
    tokenized_query = list(jieba.cut(query))
    scores = bm25.get_scores(tokenized_query)

    ranked = sorted(
        zip(scores, chunks),
        key=lambda x: x[0],
        reverse=True  # BM25 分数越高越好
    )
    return ranked[:k]  # 返回 [(bm25_score, doc), ...]


# ======================
# 3. 向量化与存储
# ======================
# 请确保你的模型路径是正确的，如果路径不对会报错
embedding_model_path = r"E:\MyOwnProj\local-rag-lab\cache\models--sentence-transformers--paraphrase-multilingual-MiniLM-L12-v2\snapshots\86741b4e3f5cb7765a600d3a3d55a0f6a6cb443d"

# 增加一个容错，如果没有本地模型，尝试从 HuggingFace 拉取（可选）
try:
    embedding = HuggingFaceEmbeddings(model_name=embedding_model_path)
except:
    print("未找到本地 Embedding 模型，尝试使用默认名称（需联网）...")
    embedding = HuggingFaceEmbeddings(model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")

# vectorstore = FAISS.from_documents(chunks, embedding)

vectorstore = FAISS.from_documents(
    chunks,
    embedding,
    distance_strategy="MAX_INNER_PRODUCT" # 切换为内积/余弦相似度
)

# Retriever 接口保留（用于单路召回或兼容其他链），但在混合检索中我们直接用 vectorstore
retriever = vectorstore.as_retriever(search_kwargs={"k": 3})


from typing import List, Tuple, Union
from langchain_core.documents import Document
from collections import defaultdict
import numpy as np

# def hybrid_retrieve(
#     query: str,
#     vec_k: int = 10,
#     bm25_k: int = 10,
#     rrf_k: int = 60,
#     min_rrf_score: float = 0.0,
# ) -> List[Document]:
#     if not query.strip():
#         return []
#
#     print(f"[Debug] Query: {query[:50]}...")
#
#     # 1. 向量召回
#     vec_raw = vectorstore.similarity_search_with_score(query, k=vec_k)
#     # vec_raw: List[Tuple[Document, float]]  float 是 distance
#     vec_results: List[Tuple[float, Document]] = [(-dist, doc) for doc, dist in vec_raw]  # 转负距离，越大越好
#
#     print(f"[Debug] Vec recall count: {len(vec_results)}")
#
#     # 2. BM25 召回
#     bm25_results = bm25_retrieve(query, k=bm25_k)
#     # 确保是 List[Tuple[float, Document]]
#     if bm25_results and not isinstance(bm25_results[0][0], (float, np.floating)):
#         # print("[Warning] BM25 返回格式异常！")
#         bm25_results = [(0.0, doc) for doc in bm25_results]  # 应急兜底
#
#     # print(f"[Debug] BM25 recall count: {len(bm25_results)}")
#
#     # 3. RRF 融合
#     rrf_scores = defaultdict(float)
#     doc_map = {}
#
#     # 向量部分
#     for rank, (score, doc) in enumerate(vec_results):
#         if not isinstance(doc, Document):
#             print(f"[Error] vec_results 中出现非 Document: {type(doc)}")
#             continue
#         content = doc.page_content
#         doc_map[content] = doc
#         rrf_scores[content] += 1 / (rrf_k + rank + 1)
#
#     # BM25 部分
#     for rank, (score, doc) in enumerate(bm25_results):
#         if not isinstance(doc, Document):
#             print(f"[Error] bm25_results 中出现非 Document: {type(doc)} → {score}")
#             continue
#         content = doc.page_content
#         if content not in doc_map:
#             doc_map[content] = doc
#         rrf_scores[content] += 1 / (rrf_k + rank + 1)
#
#     # 4. 排序 + 过滤
#     fused = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)
#     final_docs = []
#     for content, rrf_score in fused:
#         if rrf_score >= min_rrf_score and content in doc_map:
#             final_docs.append(doc_map[content])
#
#     print(f"[Debug] Final recall count after RRF: {len(final_docs)}")
#     return final_docs

def hybrid_retrieve(
        query: str,
        vec_k: int = 10,
        bm25_k: int = 10,
        rrf_k: int = 60,
        vec_score_threshold: float = 0.5,  # 向量相似度阈值（越大越好）
        min_rrf_score: float = 0.02,  # 最终融合阈值
) -> list[Document]:
    # 1. 向量召回 (此时返回的是余弦相似度分数)
    # vec_raw: List[Tuple[Document, float]]
    vec_raw = vectorstore.similarity_search_with_score(query, k=vec_k)

    vec_results = []
    for doc, score in vec_raw:
        # 余弦相似度：分数越高越好
        if score >= vec_score_threshold:
            vec_results.append((score, doc))

    # 2. BM25 召回
    bm25_results = bm25_retrieve(query, k=bm25_k)

    # 3. RRF 融合 (保持排名逻辑，RRF 对分数绝对值不敏感，只对排名敏感)
    rrf_scores = {}
    doc_map = {}

    for rank, (score, doc) in enumerate(vec_results):
        content = doc.page_content
        doc_map[content] = doc
        rrf_scores[content] = rrf_scores.get(content, 0) + 1 / (rrf_k + rank + 1)

    for rank, (score, doc) in enumerate(bm25_results):
        content = doc.page_content
        if content not in doc_map:
            doc_map[content] = doc
        rrf_scores[content] = rrf_scores.get(content, 0) + 1 / (rrf_k + rank + 1)

    # 4. 最终排序与过滤
    fused = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)
    final_docs = [doc_map[content] for content, rrf_score in fused if rrf_score >= min_rrf_score]

    return final_docs


# ======================
# Reranker 重排序
# ======================
reranker_path = r"E:\MyOwnProj\local-rag-lab\cache\models--BAAI--bge-reranker-base\snapshots\2cfc18c9415c912f9d8155881c133215df768a70"

try:
    reranker = CrossEncoder(reranker_path, device="cpu")
except:
    print("未找到本地 Reranker 模型，尝试使用默认名称（需联网）...")
    reranker = CrossEncoder("BAAI/bge-reranker-base", device="cpu")


# def rerank(query: str, docs: list[Document], top_k: int = 3):
#     if not docs:
#         return []
#
#     pairs = [(query, doc.page_content) for doc in docs]
#     scores = reranker.predict(pairs)
#
#     ranked = sorted(
#         zip(scores, docs),
#         key=lambda x: x[0],
#         reverse=True
#     )
#     return [doc for score, doc in ranked[:top_k]]

def rerank(query: str, docs: list[Document], top_k: int = 3, debug=False):
    pairs = [(query, doc.page_content) for doc in docs]
    scores = reranker.predict(pairs)

    ranked = sorted(zip(scores, docs), key=lambda x: x[0], reverse=True)

    if debug:
        print(f"\n[RERANK DEBUG] Query: {query}")
        for i, (score, doc) in enumerate(ranked):
            print(f"[{i}] score={score:.4f} | {doc.page_content[:60]}")

    return [doc for score, doc in ranked[:top_k]]



# ======================
# 5. 本地大模型
# ======================
llm = OllamaLLM(
    model="qwen2.5:14b",
    base_url="http://172.28.16.1:11434"
)


# =====================================================
# Day 2 & 3 功能模块
# =====================================================
def rewrite_query(question: str) -> str:
    prompt = f"""你是一个搜索查询优化助手。将用户问题改写为检索友好的形式。
    保留语义，不要解释。
    原始问题：{question}"""
    return llm.invoke(prompt).strip()


def route_question(question: str) -> str:
    prompt = f"""你是一个问题分类器。判断用户问题是否需要依赖“旅游知识文档”。
    只输出：RAG 或 DIRECT
    用户问题：{question}"""
    # 增加简单的容错处理，防止 LLM 废话
    res = llm.invoke(prompt).strip().upper()
    if "RAG" in res: return "RAG"
    return "DIRECT"


def is_context_valid(docs: list[Document]) -> bool:
    if not docs:
        return False
    total_length = sum(len(doc.page_content) for doc in docs)
    return total_length >= 100  # 稍微调低点阈值方便测试


# =====================================================
# 最终对外接口
# =====================================================
def ask(question: str):
    print(f"Received Question: {question}")

    # 0. 路由
    route = route_question(question)
    print(f"Router Decision: {route}")

    if route == "DIRECT":
        return llm.invoke(question)

    # 1. 改写
    rewritten = rewrite_query(question)
    print(f"Rewritten Query: {rewritten}")

    # 2. 混合召回 (Hybrid - Vector + BM25 with RRF)
    # 这里我们直接传入改写后的 query
    recall_docs = hybrid_retrieve(rewritten, vec_k=5, bm25_k=5)
    print(f"Recall Docs Count: {len(recall_docs)}")

    # 3. 重排序 (Rerank)
    final_docs = rerank(rewritten, recall_docs, top_k=3)

    # 4. 上下文校验
    if not is_context_valid(final_docs):
        return "当前知识库中没有足够信息回答该问题。"

    context = "\n\n".join(doc.page_content for doc in final_docs)

    prompt = f"""
你是一个严格基于上下文回答问题的助手。
如果上下文中没有答案，请明确说明“不知道”。

上下文：
{context}

问题：
{question}
"""
    print("Generating Answer...")
    return llm.invoke(prompt)


# ======================
# 测试
# ======================
if __name__ == "__main__":
    while True:
        try:
            q = input("\n请输入问题（exit 退出）：")
            if q.lower() in ("exit", "quit"):
                break
            print("\nAI 回答：")
            print(ask(q))
            print("=" * 60)
        except Exception as e:
            print(f"发生错误: {e}")
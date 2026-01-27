from langchain_ollama import OllamaLLM
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from rank_bm25 import BM25Okapi
from sentence_transformers import CrossEncoder
from collections import OrderedDict
import jieba

# ======================
# 1. 读取文档
# ======================
with open("data/tourism_dpo.md", "r", encoding="utf-8") as f:
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
print("chunks=",chunks)



# 所有 chunk 文本
bm25_corpus = [doc.page_content for doc in chunks]

# 简单分词（中文可先用空格，下面的代码是升级为用jieba）
# bm25_tokenized = [text.split() for text in bm25_corpus]
bm25_tokenized = [list(jieba.cut(text)) for text in bm25_corpus]  # 关键升级

bm25 = BM25Okapi(bm25_tokenized)

# 用 BM25 检索 Top-N
def bm25_retrieve(query: str, k: int = 5):
    # tokenized_query = query.split()
    tokenized_query = list(jieba.cut(query))
    scores = bm25.get_scores(tokenized_query)

    ranked = sorted(
        zip(scores, chunks),
        key=lambda x: x[0],
        reverse=True
    )

    return [doc for score, doc in ranked[:k]]


#和向量召回合并
# def hybrid_retrieve(query: str):
#     vec_docs = retriever.invoke(query)
#     bm25_docs = bm25_retrieve(query)
#
#     all_docs = vec_docs + bm25_docs
#
#     # 去重
#     unique_docs = OrderedDict()
#     for doc in all_docs:
#         unique_docs[doc.page_content] = doc
#
#     return list(unique_docs.values())
from langchain_core.documents import Document
#参照ragflow的融合排序RRF
def hybrid_retrieve(
        query: str,
        vec_k: int = 10,  # 向量取 top
        bm25_k: int = 10,  # BM25 取 top
        rrf_k: int = 60  # RRF 常数，越大越平滑（RAGFlow 默认类似 60）
) -> list[Document]:
    # 1. 分别召回带排名
    vec_results = vectorstore.similarity_search_with_score(query, k=vec_k)  # 假设支持分数；或手动算
    # vec_results: [(similarity_score, doc), ...]  similarity 越高越好

    bm25_results = bm25_retrieve(query, k=bm25_k)  # [(bm25_score, doc), ...]

    # 2. RRF 融合
    scores = {}
    for rank, (_, doc) in enumerate(vec_results):
        content = doc.page_content
        scores[content] = scores.get(content, 0) + 1 / (rrf_k + rank + 1)

    for rank, (_, doc) in enumerate(bm25_results):
        content = doc.page_content
        scores[content] = scores.get(content, 0) + 1 / (rrf_k + rank + 1)

    # 3. 排序 + 去重（按 content）
    fused = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    unique_docs = OrderedDict()
    for content, _ in fused:
        # 找原 doc（从 vec 或 bm25 取一个）
        for _, doc in vec_results + bm25_results:
            if doc.page_content == content:
                unique_docs[content] = doc
                break

    return list(unique_docs.values())



reranker = CrossEncoder(
    r"E:\MyOwnProj\local-rag-lab\cache\models--BAAI--bge-reranker-base\snapshots\2cfc18c9415c912f9d8155881c133215df768a70",
    device="cpu"  # 或 cuda
)

def rerank(query: str, docs: list[Document], top_k: int = 3):
    pairs = [(query, doc.page_content) for doc in docs]

    scores = reranker.predict(pairs)

    ranked = sorted(
        zip(scores, docs),
        key=lambda x: x[0],
        reverse=True
    )

    return [doc for score, doc in ranked[:top_k]]


# ======================
# 3. 向量化（baseline，不动）
# ======================
embedding = HuggingFaceEmbeddings(
    model_name=r"E:\MyOwnProj\local-rag-lab\cache\models--sentence-transformers--paraphrase-multilingual-MiniLM-L12-v2\snapshots\86741b4e3f5cb7765a600d3a3d55a0f6a6cb443d"
)

vectorstore = FAISS.from_documents(chunks, embedding)

# ======================
# 4. Retriever
# ======================
retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
# ======================
# 5. 本地大模型
# ======================
llm = OllamaLLM(
    model="qwen2.5:14b",
    base_url="http://172.28.16.1:11434"
)

# =====================================================
# Day 2 新增能力 ①：Query Rewrite
# =====================================================
def rewrite_query(question: str) -> str:
    prompt = f"""
你是一个搜索查询优化助手。
你的任务是将用户问题改写为更适合在旅游知识文档中检索的形式。

要求：
1. 保留原始语义
2. 使用偏书面、信息检索友好的表达
3. 不引入文档中可能不存在的信息
4. 只输出改写后的问题，不要解释

原始问题：
{question}
"""
    return llm.invoke(prompt).strip()


# =====================================================
# Day 2 新增能力 ②：Multi-Query Expansion
# =====================================================
def generate_multi_queries(question: str, n: int = 3) -> list[str]:
    prompt = f"""
你是一个检索查询生成器。
请基于下面的问题，生成 {n} 个不同表达方式但语义一致的检索查询。

要求：
1. 每个查询侧重点略有不同
2. 使用适合旅游知识文档检索的表达
3. 不引入新事实
4. 每行一个查询

问题：
{question}
"""
    result = llm.invoke(prompt)
    queries = [line.strip("- ").strip()
               for line in result.split("\n")
               if line.strip()]
    return queries


# =====================================================
# Day 2 新增能力 ③：多查询检索 + 去重
# =====================================================
def retrieve_with_multi_query(queries: list[str]):
    all_docs = []

    for q in queries:
        docs = retriever.invoke(q)
        all_docs.extend(docs)

    # 基于内容去重
    unique_docs = OrderedDict()
    for doc in all_docs:
        unique_docs[doc.page_content] = doc

    return list(unique_docs.values())




# =====================================================
# Day 3 新增：Query Router（是否需要 RAG）
# =====================================================
def route_question(question: str) -> str:
    prompt = f"""
你是一个问题分类器。
请判断用户问题是否需要依赖“旅游知识文档”才能回答。

只允许输出以下两种之一：
- RAG
- DIRECT

用户问题：
{question}
"""
    result = llm.invoke(prompt).strip()
    return result


# =====================================================
# Day 3 新增：RAG 降级判断。如果匹配到的chunk总数小于200，说明知识库里关于这方面内容的信息很少，
# =====================================================
def is_context_valid(docs: list[Document]) -> bool:
    if not docs:
        return False

    total_length = sum(len(doc.page_content) for doc in docs)
    return total_length >= 200  # 经验阈值，可调




# ======================
# 最终 RAG 接口
# ======================
def ask(question: str):
    route = route_question(question)

    if route == "DIRECT":
        return llm.invoke(question)

    rewritten = rewrite_query(question)

    # ① 多路召回
    recall_docs = hybrid_retrieve(rewritten)

    # ② Rerank
    final_docs = rerank(rewritten, recall_docs)

    # ③ 上下文校验
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
    return llm.invoke(prompt)



# ======================
# 测试
# ======================
if __name__ == "__main__":
    while True:
        q = input("请输入问题（exit 退出）：")
        if q.lower() in ("exit", "quit"):
            break
        print("\nAI 回答：")
        print(ask(q))
        print("=" * 60)
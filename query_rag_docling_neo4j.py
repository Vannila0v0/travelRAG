from langchain_ollama import OllamaLLM
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
from rank_bm25 import BM25Okapi
from sentence_transformers import CrossEncoder
from collections import OrderedDict
import jieba
import os
from etl_parser import parse_pdf_to_markdown, chunk_markdown
from core.neo4j_handler import neo4j_client

# ======================
# 1. 深度文档解析 (Deep Document Understanding)
# ======================
# 假设我们要处理 data 目录下的 PDF
data_path = "../data/桂林旅游产品常用知识(1).docx"

# 判断是读缓存还是重新解析（真实项目中通常会把解析结果存数据库）
if os.path.exists(data_path):
    # 使用 Docling 进行视觉解析，保留表格结构
    print("正在使用视觉模型解析...")
    raw_markdown = parse_pdf_to_markdown(data_path)

    # 使用 Markdown 结构化切分，防止把表格切碎
    chunks = chunk_markdown(raw_markdown)

    print(f"文档加载完成，共切分为 {len(chunks)} 个语义块。")
else:
    # 降级处理：如果没有 PDF，还是读原来的 MD
    print("未找到 PDF，降级加载 text...")
    with open("../data/tourism_dpo.md", "r", encoding="utf-8") as f:
        text = f.read()
    # ... 原来的切分逻辑 ...
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = splitter.create_documents([text])


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
        vec_k: int = 10,
        bm25_k: int = 10,
        rrf_k: int = 60
) -> list[Document]:
    # 1. 向量召回
    # FAISS 返回格式：[(Document, score), (Document, score), ...]
    vec_results = vectorstore.similarity_search_with_score(query, k=vec_k)

    # 2. BM25 召回
    # 你的 bm25_retrieve 返回的是 [Document, Document, ...] (根据你之前提供的代码)
    bm25_results = bm25_retrieve(query, k=bm25_k)

    # 3. RRF 融合计算
    scores = {}

    # 处理向量结果：解包顺序修正为 (doc, score)
    for rank, (doc, score) in enumerate(vec_results):
        content = doc.page_content
        # RRF 倒数排名融合算法
        if content not in scores:
            scores[content] = 0
        scores[content] += 1 / (rrf_k + rank + 1)

    # 处理 BM25 结果：直接遍历 Document
    for rank, doc in enumerate(bm25_results):
        content = doc.page_content
        if content not in scores:
            scores[content] = 0
        scores[content] += 1 / (rrf_k + rank + 1)

    # 4. 排序 + 重建文档列表
    # 按照 RRF 得分从高到低排序
    fused = sorted(scores.items(), key=lambda x: x[1], reverse=True)

    final_docs = []
    # 建立一个临时映射表 {content: doc} 以便根据 content 找回 doc 对象
    # 优先使用 vectorstore 里的 doc (包含 metadata)，如果没有则用 bm25 的
    content_to_doc = {}

    for doc, _ in vec_results:
        content_to_doc[doc.page_content] = doc

    for doc in bm25_results:
        if doc.page_content not in content_to_doc:
            content_to_doc[doc.page_content] = doc

    # 生成最终结果
    for content, score in fused:
        if content in content_to_doc:
            final_docs.append(content_to_doc[content])

    return final_docs


# 2. 新增：基于图谱的查询扩展函数
def graph_query_expansion(query_str: str) -> str:
    """
    输入用户问题，利用 Neo4j 扩展相关实体
    """
    print(f"正在进行图谱扩展分析: {query_str} ...")

    # 这里为了简单，用 jieba 简单分词找名词作为“潜在实体”
    # 更好的做法是用 LLM 提取问题中的实体 (NER)
    import jieba.posseg as pseg
    words = pseg.cut(query_str)
    potential_entities = [w.word for w in words if w.flag.startswith("n")]  # 取名词

    expanded_keywords = set()

    for entity in potential_entities:
        # A. 先去图里模糊查一下有没有这个点 (比如用户搜"迪士尼", 库里是"上海迪士尼")
        real_entity = neo4j_client.fuzzy_search(entity)

        if real_entity:
            print(f"  - 命中图谱实体: {real_entity}")
            # B. 如果有，找它的邻居 (比如"上海迪士尼" -> "疯狂动物城", "创极速光轮")
            neighbors = neo4j_client.query_1hop_neighbors(real_entity, limit=3)
            expanded_keywords.update(neighbors)

    # 将扩展词拼接到原查询后面
    if expanded_keywords:
        expansion_text = " ".join(expanded_keywords)
        print(f"  - 扩展词: {expansion_text}")
        return f"{query_str} {expansion_text}"
    else:
        return query_str


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
# 3. 修改 ask 函数
def ask(question: str):
    # --- 步骤 1: 路由判断 (保留你之前的) ---
    route = route_question(question)
    if route == "DIRECT":
        return llm.invoke(question)

    # --- 步骤 2: 图谱增强 (Graph RAG) ---
    # 在改写之前或之后都可以，这里建议在原问题上做扩展，给改写提供更多上下文
    expanded_query = graph_query_expansion(question)

    # --- 步骤 3: 查询改写 ---
    # 注意：现在 rewrite 接收的是包含了扩展词的长 Query
    rewritten = rewrite_query(expanded_query)

    # --- 步骤 4: 混合检索 ---
    recall_docs = hybrid_retrieve(rewritten)

    # --- 步骤 5: 重排序 ---
    final_docs = rerank(rewritten, recall_docs)

    # ... (后续生成逻辑不变) ...
    context = "\n\n".join(doc.page_content for doc in final_docs)
    prompt = f"""
    你是一个严格基于上下文回答问题的助手。

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
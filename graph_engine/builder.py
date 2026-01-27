import sys
import os

# 将项目根目录加入路径，防止 import 报错
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.neo4j_handler import Neo4jHandler
from graph_engine.extraction import extract_graph_from_text
from etl_parser import parse_pdf_to_markdown, chunk_markdown

# 配置
NEO4J_URI = "bolt://localhost:7687"
NEO4J_USER = "neo4j"
NEO4J_PASSWORD = "password123"


def build_knowledge_graph(file_path):
    # 1. 初始化数据库连接
    neo4j = Neo4jHandler(NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD)

    # 2. 文档解析 (ETL)
    print(f"STEP 1: 解析文档 {file_path}...")

    # === 修改点开始 ===
    # Docling (DocumentConverter) 支持 PDF, DOCX, PPTX, HTML 等多种格式
    # 所以只要不是纯文本(.txt/.md)，都应该走 parse_pdf_to_markdown (建议改名为 parse_doc_to_markdown)
    lower_path = file_path.lower()

    if lower_path.endswith((".pdf", ".docx", ".doc", ".pptx", ".html")):
        print(f"   检测到复杂文档格式，正在调用 Docling 进行深度解析...")
        # 虽然函数名叫 parse_pdf_... 但底层 DocumentConverter 支持 docx
        md_text = parse_pdf_to_markdown(file_path)
        chunks = chunk_markdown(md_text)
    else:
        # 兜底：处理 .txt, .md 或其他纯文本
        print(f"   检测为纯文本格式...")
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()

        # 简单封装，防止报错
        from langchain_core.documents import Document
        # 这里建议加上切分逻辑，否则长文本会撑爆 LLM 上下文
        from langchain_text_splitters import RecursiveCharacterTextSplitter
        splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
        chunks = splitter.create_documents([content])
    # === 修改点结束 ===

    print(f"文档切分完成，共 {len(chunks)} 个片段。")

    # 3. 循环提取并写入 (Write Path)
    for i, chunk in enumerate(chunks):
        content = chunk.page_content
        if len(content) < 50: continue  # 跳过过短的

        print(f"STEP 2: 处理 Chunk {i + 1}/{len(chunks)} ...")

        # A. 提取 (Extraction)
        graph_data = extract_graph_from_text(content)

        print(f"   -> 提取到 {len(graph_data.entities)} 实体, {len(graph_data.relationships)} 关系")

        # B. 写入 (Storage)
        if graph_data.entities:
            neo4j.add_graph_data(graph_data.entities, graph_data.relationships)

    print("图谱构建完成！")
    neo4j.close()


if __name__ == "__main__":
    # 你的 docx 文件路径
    target_file = "../data/桂林旅游产品常用知识(1).docx"

    if os.path.exists(target_file):
        build_knowledge_graph(target_file)
    else:
        print(f"错误：找不到文件 {target_file}")
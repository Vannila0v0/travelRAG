from docling.document_converter import DocumentConverter
from langchain_core.documents import Document
from langchain_text_splitters import MarkdownHeaderTextSplitter


def parse_pdf_to_markdown(file_path: str):
    """
    深度文档解析：使用视觉模型识别表格和布局，转为结构化 Markdown
    """
    print(f"正在深度解析文档: {file_path} ...")

    # 1. 初始化转换器 (这是 RAGFlow DeepDoc 的核心替代品)
    converter = DocumentConverter()

    # 2. 转换 (支持 PDF, PPTX, DOCX 等)
    result = converter.convert(file_path)

    # 3. 导出为 Markdown (此时表格会被转为标准 MD 格式，而不是乱码)
    full_markdown = result.document.export_to_markdown()

    return full_markdown


def chunk_markdown(markdown_text: str):
    """
    结构化切分：基于 Markdown 标题切分，而不是生硬地按字符数切分
    """
    # 定义标题层级，确保切分出来的块包含上下文
    headers_to_split_on = [
        ("#", "Header 1"),
        ("##", "Header 2"),
        ("###", "Header 3"),
    ]

    markdown_splitter = MarkdownHeaderTextSplitter(
        headers_to_split_on=headers_to_split_on,
        strip_headers=False  # 保留标题在正文中，增强语义
    )

    md_header_splits = markdown_splitter.split_text(markdown_text)

    # 如果某个章节特别长，再进行二次字符切分，但通常 MD 切分已经足够好
    return md_header_splits


# 测试代码
if __name__ == "__main__":
    # 找一个带表格的 PDF 测试
    md = parse_pdf_to_markdown("data/桂林旅游产品常用知识(1).docx")
    chunks = chunk_markdown(md)
    print(f"解析得到 {len(chunks)} 个语义块")
    print("第一个块内容示例：\n", chunks[0].page_content)
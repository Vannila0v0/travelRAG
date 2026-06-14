import hashlib
from pathlib import Path

from etl_parser import chunk_markdown, parse_pdf_to_markdown


STRUCTURED_EXTENSIONS = {".pdf", ".docx", ".pptx", ".html", ".xlsx"}
TEXT_EXTENSIONS = {".txt", ".md"}
SUPPORTED_EXTENSIONS = STRUCTURED_EXTENSIONS | TEXT_EXTENSIONS


def make_doc_id(file_path: str) -> str:
    path = Path(file_path).resolve()
    digest = hashlib.sha1(str(path).lower().encode("utf-8")).hexdigest()[:10]
    return f"{path.stem}-{digest}"


def make_chunk_id(doc_id: str, chunk_index: int) -> str:
    return f"{doc_id}#chunk-{chunk_index:05d}"


def normalize_metadata(metadata) -> dict:
    if not metadata:
        return {}
    return {str(key): str(value) for key, value in dict(metadata).items()}


def metadata_section(metadata: dict) -> str | None:
    for key in ("Header 3", "Header 2", "Header 1", "section", "title"):
        value = metadata.get(key)
        if value:
            return str(value)
    return None


def metadata_page(metadata: dict) -> int | None:
    for key in ("page", "page_no", "page_number"):
        value = metadata.get(key)
        if value is None:
            continue
        try:
            return int(value)
        except ValueError:
            return None
    return None


def load_and_chunk_document(file_path: str):
    path = Path(file_path)
    suffix = path.suffix.lower()

    if suffix in STRUCTURED_EXTENSIONS:
        print(f"   Detected structured document, parsing with Docling: {file_path}")
        markdown_text = parse_pdf_to_markdown(file_path)
        return chunk_markdown(markdown_text)

    if suffix in TEXT_EXTENSIONS:
        print(f"   Detected text document: {file_path}")
        with path.open("r", encoding="utf-8") as f:
            content = f.read()

        from langchain_text_splitters import RecursiveCharacterTextSplitter

        splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
        return splitter.create_documents([content])

    raise ValueError(f"Unsupported file extension: {suffix}")

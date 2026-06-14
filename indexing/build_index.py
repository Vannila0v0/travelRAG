import json
from dataclasses import asdict, dataclass
from pathlib import Path

from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document

from agent_system.integration.llm_factory import get_embeddings_model
from document_pipeline import (
    load_and_chunk_document,
    make_chunk_id,
    make_doc_id,
    metadata_page,
    metadata_section,
    normalize_metadata,
)


@dataclass
class IndexStats:
    files_attempted: int = 0
    files_failed: int = 0
    chunks: int = 0
    index_dir: str = ""


def prepare_index_documents(files: list[Path], min_chunk_chars: int = 20) -> tuple[list[Document], list[tuple[Path, str]]]:
    documents: list[Document] = []
    failures: list[tuple[Path, str]] = []

    for file_path in files:
        try:
            doc_id = make_doc_id(str(file_path))
            source_path = str(file_path.resolve())
            chunks = load_and_chunk_document(str(file_path))

            for index, chunk in enumerate(chunks):
                content = chunk.page_content
                if len(content.strip()) < min_chunk_chars:
                    continue

                metadata = normalize_metadata(getattr(chunk, "metadata", {}))
                chunk_id = make_chunk_id(doc_id, index)
                documents.append(
                    Document(
                        page_content=content,
                        metadata={
                            **metadata,
                            "doc_id": doc_id,
                            "chunk_id": chunk_id,
                            "chunk_index": index,
                            "source": source_path,
                            "source_path": source_path,
                            "file_name": file_path.name,
                            "page": metadata_page(metadata),
                            "section": metadata_section(metadata),
                        },
                    )
                )
        except Exception as exc:
            failures.append((file_path, str(exc)))

    return documents, failures


def build_faiss_index(files: list[Path], index_dir: Path, min_chunk_chars: int = 20) -> tuple[IndexStats, list[tuple[Path, str]]]:
    documents, failures = prepare_index_documents(files, min_chunk_chars=min_chunk_chars)
    stats = IndexStats(
        files_attempted=len(files),
        files_failed=len(failures),
        chunks=len(documents),
        index_dir=str(index_dir.resolve()),
    )

    if not documents:
        return stats, failures

    index_dir.mkdir(parents=True, exist_ok=True)
    embeddings = get_embeddings_model()
    vectorstore = FAISS.from_documents(documents, embeddings)
    vectorstore.save_local(str(index_dir))

    manifest = {
        **asdict(stats),
        "sources": sorted({doc.metadata["source_path"] for doc in documents}),
    }
    (index_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    return stats, failures

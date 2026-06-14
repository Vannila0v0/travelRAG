import os
import sys
from dataclasses import dataclass
from pathlib import Path

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.config import neo4j_password, neo4j_url, neo4j_user
from core.neo4j_handler import Neo4jHandler
from document_pipeline import (
    SUPPORTED_EXTENSIONS,
    load_and_chunk_document,
    make_chunk_id,
    make_doc_id,
    metadata_page,
    metadata_section,
    normalize_metadata,
)
from graph_engine.extraction import extract_graph_from_text
from graph_engine.schema import SourceRef


NEO4J_URI = neo4j_url
NEO4J_USER = neo4j_user
NEO4J_PASSWORD = neo4j_password

@dataclass
class BuildStats:
    file_path: str
    doc_id: str = ""
    chunks: int = 0
    processed_chunks: int = 0
    skipped_chunks: int = 0
    entities: int = 0
    relationships: int = 0


def build_knowledge_graph(file_path: str, min_chunk_chars: int = 50) -> BuildStats:
    doc_id = make_doc_id(file_path)
    source_path = str(Path(file_path).resolve())
    stats = BuildStats(file_path=file_path, doc_id=doc_id)
    neo4j = Neo4jHandler(NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD)

    try:
        print(f"STEP 1: Parsing document: {file_path}")
        chunks = load_and_chunk_document(file_path)
        stats.chunks = len(chunks)
        print(f"Document chunking completed: {stats.chunks} chunks")

        for index, chunk in enumerate(chunks):
            content = chunk.page_content
            metadata = normalize_metadata(getattr(chunk, "metadata", {}))
            chunk_id = make_chunk_id(doc_id, index)
            source_ref = SourceRef(
                doc_id=doc_id,
                chunk_id=chunk_id,
                source_path=source_path,
                chunk_index=index,
                page=metadata_page(metadata),
                section=metadata_section(metadata),
            )

            neo4j.add_document_chunk(
                doc_id=doc_id,
                doc_name=Path(file_path).name,
                source_path=source_path,
                chunk_id=chunk_id,
                chunk_index=index,
                text=content,
                metadata=metadata,
            )

            if len(content.strip()) < min_chunk_chars:
                stats.skipped_chunks += 1
                continue

            print(f"STEP 2: Processing chunk {index + 1}/{stats.chunks}")
            graph_data = extract_graph_from_text(content)

            for entity in graph_data.entities:
                entity.source_refs.append(source_ref)
            for relationship in graph_data.relationships:
                relationship.source_refs.append(source_ref)

            entity_count = len(graph_data.entities)
            relationship_count = len(graph_data.relationships)
            print(f"   -> Extracted {entity_count} entities, {relationship_count} relationships")

            if graph_data.entities:
                neo4j.add_graph_data(graph_data.entities, graph_data.relationships)

            stats.processed_chunks += 1
            stats.entities += entity_count
            stats.relationships += relationship_count

        print(f"Graph build completed for: {file_path}")
        return stats
    finally:
        neo4j.close()


if __name__ == "__main__":
    target_file = Path(__file__).resolve().parents[1] / "data" / "桂林旅游产品常用知识(1).docx"

    if target_file.exists():
        build_knowledge_graph(str(target_file))
    else:
        print(f"File not found: {target_file}")

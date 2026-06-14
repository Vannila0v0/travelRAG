import json

from .types import Source


def source_from_document(doc, score: float | None = None) -> Source:
    metadata = dict(getattr(doc, "metadata", {}) or {})
    return Source(
        doc_id=metadata.get("doc_id"),
        chunk_id=metadata.get("chunk_id"),
        source_path=metadata.get("source_path") or metadata.get("source"),
        file_name=metadata.get("file_name"),
        chunk_index=metadata.get("chunk_index"),
        page=metadata.get("page"),
        section=metadata.get("section"),
        text=getattr(doc, "page_content", None),
        score=score,
    )


def source_from_chunk_record(record) -> Source:
    metadata = {}
    metadata_json = record.get("metadata_json")
    if metadata_json:
        try:
            metadata = json.loads(metadata_json)
        except json.JSONDecodeError:
            metadata = {}
    return Source(
        doc_id=record.get("doc_id"),
        chunk_id=record.get("chunk_id"),
        source_path=record.get("source_path"),
        file_name=record.get("doc_name"),
        chunk_index=record.get("chunk_index"),
        page=metadata.get("page"),
        section=metadata.get("section"),
        text=record.get("text"),
    )


def dedupe_sources(sources: list[Source]) -> list[Source]:
    seen = set()
    result = []
    for source in sources:
        key = source.chunk_id or (source.source_path, source.chunk_index, source.text)
        if key in seen:
            continue
        seen.add(key)
        result.append(source)
    return result

from typing import List

from pydantic import BaseModel, Field


class SourceRef(BaseModel):
    doc_id: str = Field(description="Stable document id")
    chunk_id: str = Field(description="Stable chunk id within the document corpus")
    source_path: str = Field(description="Original source file path")
    chunk_index: int = Field(description="Zero-based chunk index in the document")
    page: int | None = Field(default=None, description="Page number if available")
    section: str | None = Field(default=None, description="Section or heading if available")


class Entity(BaseModel):
    name: str = Field(description="Entity name")
    type: str = Field(description="Entity type")
    description: str = Field(default="", description="Entity description")
    source_refs: List[SourceRef] = Field(default_factory=list)


class Relationship(BaseModel):
    source: str = Field(description="Source entity name")
    target: str = Field(description="Target entity name")
    relation_type: str = Field(description="Relationship type")
    description: str = Field(default="", description="Relationship description")
    source_refs: List[SourceRef] = Field(default_factory=list)


class GraphResult(BaseModel):
    entities: List[Entity]
    relationships: List[Relationship]

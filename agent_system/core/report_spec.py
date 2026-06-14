from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field

class SectionOutline(BaseModel):
    """章节大纲"""
    section_id: str
    title: str
    description: str
    evidence_needs: List[str] = Field(default_factory=list, description="该章节需要的证据类型")

class ReportOutline(BaseModel):
    """报告整体大纲"""
    title: str
    report_type: str = "long_document"
    abstract: str
    sections: List[SectionOutline]

class SectionContent(BaseModel):
    """章节具体内容"""
    section_id: str
    title: str
    content: str
    citations: List[str] = Field(default_factory=list, description="本章节引用的证据ID")

class ReportResult(BaseModel):
    """最终报告产物"""
    outline: ReportOutline
    sections: List[SectionContent]
    final_report: str
    references: Optional[str] = None
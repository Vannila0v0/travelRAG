from datetime import datetime
from typing import Any, Dict, List, Optional
import uuid

from pydantic import BaseModel, Field


class ExecutionMetadata(BaseModel):
    worker_type: str
    latency_seconds: float = 0.0
    tool_calls_count: int = 0
    error: Optional[str] = None


class ExecutionRecord(BaseModel):
    """Record one executed subtask and the evidence returned by its tool."""

    record_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    task_id: str
    worker_type: str
    inputs: Dict[str, Any]
    output: Any
    metadata: ExecutionMetadata
    route: Optional[str] = None
    sources: List[Dict[str, Any]] = Field(default_factory=list)
    tool_metadata: Dict[str, Any] = Field(default_factory=dict)
    timestamp: datetime = Field(default_factory=datetime.now)

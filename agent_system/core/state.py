from datetime import datetime
from typing import Any, List, Optional

from pydantic import BaseModel, Field

from .execution_record import ExecutionRecord
from .plan_spec import PlanSpec


class PlanExecuteState(BaseModel):
    """Shared state for Plan-Execute-Report orchestration."""

    session_id: str
    input_query: str

    plan: Optional[PlanSpec] = None

    execution_records: List[ExecutionRecord] = Field(default_factory=list)
    completed_task_ids: List[str] = Field(default_factory=list)
    sources: List[dict[str, Any]] = Field(default_factory=list)

    final_report: Optional[str] = None

    created_at: datetime = Field(default_factory=datetime.now)
    updated_at: datetime = Field(default_factory=datetime.now)

    def update_timestamp(self):
        self.updated_at = datetime.now()

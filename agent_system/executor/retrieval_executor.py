import time
import logging
from typing import Any, Dict
from ..core.execution_record import ExecutionRecord, ExecutionMetadata
from ..core.plan_spec import TaskNode
from ..core.state import PlanExecuteState
from .tool_registry import TOOL_REGISTRY

_LOGGER = logging.getLogger(__name__)


class RetrievalExecutor:
    """检索任务执行器"""

    def can_handle(self, task_type: str) -> bool:
        return task_type in TOOL_REGISTRY

    def execute_task(self, task: TaskNode, state: PlanExecuteState) -> ExecutionRecord:
        """执行单个任务"""
        tool_name = task.task_type

        # 1. 获取工具
        if tool_name not in TOOL_REGISTRY:
            raise ValueError(f"未知工具类型: {tool_name}")

        tool_instance = TOOL_REGISTRY[tool_name]()

        # 2. 准备参数
        payload = task.parameters.copy()
        payload["description"] = task.description
        payload["entities"] = getattr(task, "entities", [])
        payload["task_type"] = task.task_type

        # 3. 执行工具
        start_time = time.perf_counter()
        try:
            result = tool_instance.structured_search(payload)
            output = result.get("answer", "")
            data = result.get("data", {}) or {}
            route = data.get("route")
            sources = data.get("sources", [])
            tool_metadata = data.get("metadata", {}) or {}
            error = result.get("error")
        except Exception as e:
            output = None
            route = None
            sources = []
            tool_metadata = {}
            error = str(e)

        latency = time.perf_counter() - start_time

        # 4. 生成记录
        metadata = ExecutionMetadata(
            worker_type="retrieval_executor",
            latency_seconds=latency,
            error=error
        )

        record = ExecutionRecord(
            task_id=task.task_id,
            worker_type="retrieval_executor",
            inputs=payload,
            output=output,
            route=route,
            sources=sources,
            tool_metadata=tool_metadata,
            metadata=metadata
        )

        return record

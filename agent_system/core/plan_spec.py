from typing import List, Dict, Any, Optional, Literal
from datetime import datetime
import uuid
from pydantic import BaseModel, Field

# 定义支持的任务类型，可以扩展
TaskTypeLiteral = Literal[
    "local_search",   # 查微观实体
    "global_search",  # 查宏观概况
    "reflection",     # 反思
]

class TaskNode(BaseModel):
    """任务节点：定义单个子任务"""
    task_id: str = Field(default_factory=lambda: f"task_{uuid.uuid4().hex[:8]}")
    task_type: TaskTypeLiteral
    description: str
    priority: int = Field(default=2, description="1=High, 2=Medium, 3=Low")
    depends_on: List[str] = Field(default_factory=list, description="依赖的任务ID列表")
    entities: List[str] = Field(default_factory=list)
    parameters: Dict[str, Any] = Field(default_factory=dict)
    status: Literal["pending", "running", "completed", "failed"] = "pending"
    result: Optional[str] = None # 用于简易存储结果

class TaskGraph(BaseModel):
    """任务图：定义任务及其依赖关系"""
    nodes: List[TaskNode]
    execution_mode: Literal["sequential", "parallel"] = "sequential"

    def get_ready_tasks(self, completed_task_ids: List[str]) -> List[TaskNode]:
        """获取当前依赖已满足，可以执行的任务"""
        ready = []
        for node in self.nodes:
            if node.status != "pending":
                continue
            # 检查依赖是否全部在已完成列表中
            if all(dep in completed_task_ids for dep in node.depends_on):
                ready.append(node)
        return sorted(ready, key=lambda x: x.priority)

class PlanSpec(BaseModel):
    """完整的计划规范"""
    plan_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    original_query: str
    task_graph: TaskGraph
    created_at: datetime = Field(default_factory=datetime.now)
    status: Literal["draft", "executing", "completed", "failed"] = "draft"

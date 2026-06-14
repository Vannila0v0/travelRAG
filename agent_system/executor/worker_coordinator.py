import logging
import os
from concurrent.futures import ThreadPoolExecutor, as_completed

from ..core.execution_record import ExecutionRecord
from ..core.plan_spec import TaskNode
from ..core.state import PlanExecuteState
from .retrieval_executor import RetrievalExecutor


_LOGGER = logging.getLogger(__name__)


def _max_workers() -> int:
    try:
        return max(1, int(os.getenv("AGENT_MAX_WORKERS", "3")))
    except ValueError:
        return 3


class WorkerCoordinator:
    """Schedule DAG tasks and run independent ready tasks concurrently."""

    def __init__(self, max_workers: int | None = None):
        self.executors = [RetrievalExecutor()]
        self.max_workers = max_workers or _max_workers()

    def run(self, state: PlanExecuteState):
        if not state.plan:
            return

        task_graph = state.plan.task_graph
        completed_ids: set[str] = set(state.completed_task_ids)

        while len(completed_ids) < len(task_graph.nodes):
            ready_tasks = task_graph.get_ready_tasks(list(completed_ids))

            if not ready_tasks:
                _LOGGER.error("No executable tasks found; the task graph may contain failed or blocked dependencies")
                break

            batch_size = min(self.max_workers, len(ready_tasks))
            _LOGGER.info("Executing %s ready task(s), max_workers=%s", len(ready_tasks), batch_size)

            if batch_size == 1:
                record = self._execute_single_task(ready_tasks[0], state)
                if record:
                    state.execution_records.append(record)
                if ready_tasks[0].status == "completed":
                    completed_ids.add(ready_tasks[0].task_id)
                    state.completed_task_ids = list(completed_ids)
                continue

            with ThreadPoolExecutor(max_workers=batch_size) as pool:
                future_to_task = {
                    pool.submit(self._execute_single_task, task, state): task
                    for task in ready_tasks
                }
                for future in as_completed(future_to_task):
                    task = future_to_task[future]
                    try:
                        record = future.result()
                    except Exception as exc:
                        _LOGGER.error("Task execution crashed: %s [%s]: %s", task.task_id, task.task_type, exc)
                        task.status = "failed"
                        continue

                    if record:
                        state.execution_records.append(record)
                    if task.status == "completed":
                        completed_ids.add(task.task_id)
                        state.completed_task_ids = list(completed_ids)

    def _execute_single_task(self, task: TaskNode, state: PlanExecuteState) -> ExecutionRecord | None:
        task.status = "running"
        _LOGGER.info("Start task: %s [%s]", task.task_id, task.task_type)

        executor = next((item for item in self.executors if item.can_handle(task.task_type)), None)
        if not executor:
            _LOGGER.error("No executor found for task type: %s", task.task_type)
            task.status = "failed"
            return None

        try:
            record = executor.execute_task(task, state)
            if record.metadata.error:
                task.status = "failed"
                task.result = f"Error: {record.metadata.error}"
            else:
                task.status = "completed"
                task.result = str(record.output)[:100] + "..."
            return record
        except Exception as exc:
            _LOGGER.error("Task failed: %s [%s]: %s", task.task_id, task.task_type, exc)
            task.status = "failed"
            return None

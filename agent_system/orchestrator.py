import logging
from typing import Any

from .core.plan_spec import PlanSpec
from .core.state import PlanExecuteState
from .executor.worker_coordinator import WorkerCoordinator
from .planner.task_decomposer import TaskDecomposer
from .reporter.base_reporter import BaseReporter


logging.basicConfig(level=logging.INFO)
_LOGGER = logging.getLogger(__name__)


class MultiAgentOrchestrator:
    def __init__(self):
        self.planner = TaskDecomposer()
        self.worker = WorkerCoordinator()
        self.reporter = BaseReporter()

    def run(self, query: str) -> PlanExecuteState:
        state = PlanExecuteState(
            session_id="session_1",
            input_query=query,
        )
        _LOGGER.info("[Orchestrator] start: %s", query)

        _LOGGER.info("--- Phase 1: Planning ---")
        try:
            task_graph = self.planner.decompose(state.input_query)
        except Exception as exc:
            _LOGGER.error("Planner failed: %s", exc)
            return state

        state.plan = PlanSpec(
            original_query=state.input_query,
            task_graph=task_graph,
        )

        if not state.plan or not state.plan.task_graph.nodes:
            _LOGGER.error("Planning failed: no tasks generated")
            return state

        _LOGGER.info("Planning succeeded, generated %s tasks", len(state.plan.task_graph.nodes))
        for node in state.plan.task_graph.nodes:
            _LOGGER.info("   - [%s] %s", node.task_type, node.description)

        _LOGGER.info("--- Phase 2: Execution ---")
        self.worker.run(state)
        state.sources = self._collect_sources(state)

        _LOGGER.info("--- Phase 3: Reporting ---")
        self.reporter.generate(state)

        _LOGGER.info("Workflow finished")
        return state

    @staticmethod
    def _collect_sources(state: PlanExecuteState) -> list[dict[str, Any]]:
        seen = set()
        sources: list[dict[str, Any]] = []
        for record in state.execution_records:
            for source in record.sources:
                if not isinstance(source, dict):
                    continue
                key = (
                    source.get("chunk_id")
                    or (source.get("doc_id"), source.get("source_path"), source.get("file_name"))
                    or tuple(sorted((k, str(v)) for k, v in source.items() if v is not None))
                )
                if key in seen:
                    continue
                seen.add(key)
                sources.append(source)
        return sources

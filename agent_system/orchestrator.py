import logging
import time
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

    def run(
        self,
        query: str,
        report_mode: str = "concise",
        plan_mode: str = "auto",
    ) -> PlanExecuteState:
        plan_mode = plan_mode if plan_mode in {"detailed_itinerary", "place_recommendations"} else "auto"
        state = PlanExecuteState(
            session_id="session_1",
            input_query=query,
            plan_mode=plan_mode,
        )
        report_mode = "full" if report_mode == "full" else "concise"
        state.agent_trace = {
            "planner_latency_ms": None,
            "execution_latency_ms": None,
            "reporter_latency_ms": None,
            "reporter_mode": report_mode,
            "plan_mode": plan_mode,
            "reporter_llm_calls": 0,
            "task_count": 0,
        }
        _LOGGER.info("[Orchestrator] start: %s", query)

        _LOGGER.info("--- Phase 1: Planning ---")
        phase_started = time.perf_counter()
        try:
            task_graph = self.planner.decompose(state.input_query, plan_mode=plan_mode)
        except Exception as exc:
            state.agent_trace["planner_latency_ms"] = int((time.perf_counter() - phase_started) * 1000)
            state.agent_trace["planner_error"] = str(exc)
            _LOGGER.error("Planner failed: %s", exc)
            return state
        state.agent_trace["planner_latency_ms"] = int((time.perf_counter() - phase_started) * 1000)
        state.agent_trace["plan_compaction"] = getattr(self.planner, "last_compaction_trace", {})

        state.plan = PlanSpec(
            original_query=state.input_query,
            task_graph=task_graph,
        )
        state.agent_trace["task_count"] = len(state.plan.task_graph.nodes)

        if not state.plan or not state.plan.task_graph.nodes:
            _LOGGER.error("Planning failed: no tasks generated")
            return state

        _LOGGER.info("Planning succeeded, generated %s tasks", len(state.plan.task_graph.nodes))
        for node in state.plan.task_graph.nodes:
            _LOGGER.info("   - [%s] %s", node.task_type, node.description)

        _LOGGER.info("--- Phase 2: Execution ---")
        phase_started = time.perf_counter()
        self.worker.run(state)
        state.agent_trace["execution_latency_ms"] = int((time.perf_counter() - phase_started) * 1000)
        state.sources = self._collect_sources(state)

        _LOGGER.info("--- Phase 3: Reporting ---")
        phase_started = time.perf_counter()
        self.reporter.generate(state, mode=report_mode)
        state.agent_trace["reporter_latency_ms"] = int((time.perf_counter() - phase_started) * 1000)
        state.agent_trace["reporter_mode"] = report_mode
        state.agent_trace["reporter_llm_calls"] = self.reporter.last_llm_call_count
        state.agent_trace.update(self.reporter.last_metrics)

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

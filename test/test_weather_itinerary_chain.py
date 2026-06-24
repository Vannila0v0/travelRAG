import os
import unittest
from datetime import date, timedelta
from types import SimpleNamespace
from unittest.mock import patch

from agent_system.core.plan_spec import PlanSpec
from agent_system.core.state import PlanExecuteState
from agent_system.executor.worker_coordinator import WorkerCoordinator
from agent_system.planner.task_template_matcher import TaskTemplateMatcher


class FakeLLM:
    def invoke(self, prompt):
        return SimpleNamespace(content="Weather-aware itinerary evidence is sufficient.")


class FakeQueryEngine:
    def ask(self, query, route="auto"):
        return SimpleNamespace(
            answer=f"{route} answer for {query}",
            route=route,
            sources=[],
            metadata={},
        )


class WeatherItineraryChainTest(unittest.TestCase):
    def tearDown(self):
        for key in [
            "WEATHER_QUERY_PROVIDER",
            "AGENT_TOOL_CACHE_ENABLED",
            "AGENT_MAX_WORKERS",
        ]:
            os.environ.pop(key, None)

    def test_weather_itinerary_template_runs_worker_chain(self):
        os.environ["WEATHER_QUERY_PROVIDER"] = "mock"
        os.environ["AGENT_TOOL_CACHE_ENABLED"] = "false"
        os.environ["AGENT_MAX_WORKERS"] = "1"
        start = (date.today() + timedelta(days=2)).isoformat()
        query = f"请帮我规划{start}到桂林的三日游路线安排"
        graph = TaskTemplateMatcher().match(query)
        self.assertIsNotNone(graph)

        state = PlanExecuteState(session_id="s", input_query=query)
        state.plan = PlanSpec(original_query=query, task_graph=graph)

        with (
            patch("agent_system.executor.tool_registry.get_query_engine", return_value=FakeQueryEngine()),
            patch("agent_system.executor.tool_registry.get_llm_model", return_value=FakeLLM()),
        ):
            WorkerCoordinator(max_workers=1).run(state)

        self.assertEqual(
            [record.inputs["task_type"] for record in state.execution_records],
            ["weather_query", "global_search", "local_search", "map_route", "reflection"],
        )
        self.assertEqual(
            [record.route for record in state.execution_records],
            ["weather_query", "global", "hybrid", "map_route", "reflection"],
        )
        self.assertTrue(state.execution_records[0].tool_metadata["forecast_available"])
        self.assertIn("route_order", state.execution_records[3].tool_metadata)


if __name__ == "__main__":
    unittest.main()

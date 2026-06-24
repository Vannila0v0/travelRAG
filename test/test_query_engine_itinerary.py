import unittest
from types import SimpleNamespace
from unittest.mock import patch

from agent_system.core.execution_record import ExecutionMetadata, ExecutionRecord
from agent_system.core.state import PlanExecuteState
from query_engine.router import QueryEngine


def make_state():
    state = PlanExecuteState(
        session_id="session-1",
        input_query="帮我规划一天桂林市区详细路线",
        plan_mode="detailed_itinerary",
    )
    state.final_report = "自然语言路线"
    state.execution_records.append(
        ExecutionRecord(
            task_id="task-1",
            worker_type="retrieval_executor",
            inputs={"task_type": "local_search"},
            output="证据",
            route="hybrid",
            sources=[{"chunk_id": "chunk-1", "file_name": "demo.md"}],
            tool_metadata={"cache_hit": False},
            metadata=ExecutionMetadata(worker_type="retrieval_executor", latency_seconds=0.1),
        )
    )
    state.sources = [{"chunk_id": "chunk-1", "file_name": "demo.md"}]
    state.agent_trace = {"planner_latency_ms": 1}
    return state


class FakeOrchestrator:
    def run(self, query, report_mode="concise", plan_mode="auto"):
        self.query = query
        self.report_mode = report_mode
        self.plan_mode = plan_mode
        return make_state()


class FakeItineraryBuilder:
    last_metrics = {"itinerary_latency_ms": 12, "itinerary_day_count": 1}

    def build(self, state):
        return {
            "days": [
                {
                    "date_label": "第 1 天",
                    "slots": [
                        {
                            "start_time": "09:00",
                            "end_time": "11:00",
                            "title": "象鼻山",
                            "activity": "游览象鼻山",
                            "source_refs": ["evidence_1"],
                        }
                    ],
                }
            ],
            "total_budget": "证据不足",
            "assumptions": [],
            "warnings": [],
        }


class QueryEngineItineraryTest(unittest.TestCase):
    def test_agent_itinerary_response_includes_structured_output(self):
        engine = QueryEngine(llm=SimpleNamespace())

        with (
            patch("agent_system.orchestrator.MultiAgentOrchestrator", return_value=FakeOrchestrator()),
            patch("agent_system.reporter.itinerary_builder.ItineraryBuilder", return_value=FakeItineraryBuilder()),
        ):
            result = engine.ask(
                "帮我规划一天桂林市区详细路线",
                route="agent",
                plan_mode="detailed_itinerary",
                response_format="itinerary",
            )

        self.assertEqual(result.route, "agent")
        self.assertEqual(result.structured_output["days"][0]["date_label"], "第 1 天")
        self.assertEqual(result.metadata["structured_output_type"], "itinerary")
        self.assertEqual(result.metadata["itinerary_metrics"]["itinerary_day_count"], 1)
        self.assertTrue(result.metadata["itinerary_validation"]["valid"])

    def test_itinerary_format_skips_non_detailed_plan_mode(self):
        engine = QueryEngine(llm=SimpleNamespace())

        with patch("agent_system.orchestrator.MultiAgentOrchestrator", return_value=FakeOrchestrator()):
            result = engine.ask(
                "推荐几个桂林好玩的地方",
                route="agent",
                plan_mode="place_recommendations",
                response_format="itinerary",
            )

        self.assertIsNone(result.structured_output)
        self.assertIn("structured_output_skipped_reason", result.metadata)


if __name__ == "__main__":
    unittest.main()

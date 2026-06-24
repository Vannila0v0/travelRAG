import unittest
import json
from types import SimpleNamespace

from agent_system.core.execution_record import ExecutionMetadata, ExecutionRecord
from agent_system.core.state import PlanExecuteState
from agent_system.reporter.base_reporter import BaseReporter


class FakeLLM:
    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = []

    def invoke(self, prompt):
        self.calls.append(prompt)
        return SimpleNamespace(content=self.responses.pop(0))


def make_state():
    state = PlanExecuteState(session_id="session-1", input_query="帮我规划龙胜温泉一日游")
    state.execution_records.append(
        ExecutionRecord(
            task_id="task-1",
            worker_type="retrieval_executor",
            inputs={"task_type": "local_search"},
            output="龙胜温泉可从龙胜县城汽车总站乘车前往，门票有学生和老人优惠。",
            route="hybrid",
            sources=[],
            tool_metadata={"cache_hit": False},
            metadata=ExecutionMetadata(worker_type="retrieval_executor", latency_seconds=0.1),
        )
    )
    return state


def make_route_state():
    state = PlanExecuteState(
        session_id="session-route",
        input_query="从桂林站出发，象鼻山和东西巷怎么安排？",
        plan_mode="detailed_itinerary",
    )
    state.execution_records.append(
        ExecutionRecord(
            task_id="task-route",
            worker_type="retrieval_executor",
            inputs={"task_type": "map_route", "start_time": "09:00"},
            output=json.dumps(
                {
                    "provider": "amap",
                    "mode": "taxi",
                    "origin": "桂林站",
                    "route_order": ["桂林站", "象鼻山", "东西巷"],
                    "legs": [
                        {
                            "from": "桂林站",
                            "to": "象鼻山",
                            "mode": "taxi",
                            "distance_m": 2136,
                            "duration_min": 10,
                            "provider": "amap",
                        },
                        {
                            "from": "象鼻山",
                            "to": "东西巷",
                            "mode": "taxi",
                            "distance_m": 1420,
                            "duration_min": 7,
                            "provider": "amap",
                        },
                    ],
                    "total_distance_m": 3556,
                    "total_travel_time_min": 17,
                    "estimated_finish_time": "12:17",
                    "feasible": True,
                    "warnings": [],
                },
                ensure_ascii=False,
            ),
            route="map_route",
            sources=[],
            tool_metadata={"provider": "amap", "configured": True},
            metadata=ExecutionMetadata(worker_type="retrieval_executor", latency_seconds=0.1),
        )
    )
    return state


class ReporterModeTests(unittest.TestCase):
    def test_concise_mode_uses_single_llm_call(self):
        reporter = BaseReporter(llm=FakeLLM(["简短回答"]))
        state = make_state()
        state.plan_mode = "detailed_itinerary"

        result = reporter.generate(state, mode="concise")

        self.assertEqual(state.final_report, "简短回答")
        self.assertEqual(result.outline.report_type, "concise")
        self.assertEqual(reporter.last_llm_call_count, 1)
        self.assertEqual(len(reporter._llm.calls), 1)
        self.assertEqual(reporter.last_metrics["section_count"], 1)
        self.assertIn("详细路线安排", reporter._llm.calls[0])

    def test_full_mode_uses_outline_plus_batch_section_call(self):
        reporter = BaseReporter(
            llm=FakeLLM(
                [
                    '{"title":"完整报告","abstract":"摘要","sections":[{"section_id":"sec_1","title":"交通","description":"交通方案"},{"section_id":"sec_2","title":"门票","description":"票务优惠"}]}',
                    '{"sections":[{"section_id":"sec_1","title":"交通","content":"交通正文"},{"section_id":"sec_2","title":"门票","content":"门票正文"}]}',
                ]
            )
        )
        state = make_state()
        state.plan_mode = "place_recommendations"

        result = reporter.generate(state, mode="full")

        self.assertIn("# 完整报告", state.final_report)
        self.assertEqual(len(result.sections), 2)
        self.assertIn("交通正文", state.final_report)
        self.assertIn("门票正文", state.final_report)
        self.assertEqual(reporter.last_llm_call_count, 2)
        self.assertEqual(len(reporter._llm.calls), 2)
        self.assertEqual(reporter.last_metrics["section_count"], 2)
        self.assertIn("景点/项目推荐", reporter._llm.calls[0])
        self.assertIn("景点/项目推荐", reporter._llm.calls[1])

    def test_concise_mode_turns_map_route_json_into_readable_route_section(self):
        reporter = BaseReporter(llm=FakeLLM(["请按这个顺序走。"]))
        state = make_route_state()

        result = reporter.generate(state, mode="concise")

        self.assertIn("推荐顺序：桂林站 -> 象鼻山 -> 东西巷", reporter._llm.calls[0])
        self.assertIn("桂林站 -> 象鼻山：约 10 分钟", reporter._llm.calls[0])
        self.assertNotIn('"route_order"', reporter._llm.calls[0])
        self.assertIn("路线交通依据", result.final_report)
        self.assertIn("总交通：约 17 分钟", result.final_report)
        self.assertIn("桂林站 -> 象鼻山：约 10 分钟", result.final_report)


if __name__ == "__main__":
    unittest.main()

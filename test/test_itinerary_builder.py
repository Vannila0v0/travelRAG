import unittest
from types import SimpleNamespace

from agent_system.core.execution_record import ExecutionMetadata, ExecutionRecord
from agent_system.core.state import PlanExecuteState
from agent_system.reporter.itinerary_builder import ItineraryBuilder


class FakeLLM:
    def __init__(self, response):
        self.response = response
        self.calls = []

    def invoke(self, prompt):
        self.calls.append(prompt)
        return SimpleNamespace(content=self.response)


def make_state():
    state = PlanExecuteState(
        session_id="session-1",
        input_query="帮我规划一天桂林市区详细路线",
        plan_mode="detailed_itinerary",
    )
    state.final_report = "上午游览象鼻山，下午前往两江四湖，晚上夜游。"
    state.execution_records.append(
        ExecutionRecord(
            task_id="task-1",
            worker_type="retrieval_executor",
            inputs={"task_type": "local_search"},
            output="象鼻山和两江四湖适合市区一日游。",
            route="hybrid",
            sources=[],
            tool_metadata={},
            metadata=ExecutionMetadata(worker_type="retrieval_executor", latency_seconds=0.1),
        )
    )
    return state


class ItineraryBuilderTest(unittest.TestCase):
    def test_build_parses_itinerary_json(self):
        llm = FakeLLM(
            """
            {
              "days": [
                {
                  "date_label": "第 1 天",
                  "slots": [
                    {
                      "start_time": "09:00",
                      "end_time": "11:00",
                      "title": "象鼻山",
                      "location": "桂林市区",
                      "activity": "游览象鼻山",
                      "transport_to_next": "打车前往两江四湖",
                      "estimated_cost": "以实际票价为准",
                      "ticket_info": "证据不足",
                      "source_refs": ["evidence_1"],
                      "notes": "避开高峰"
                    }
                  ]
                }
              ],
              "total_budget": "以实际票价为准",
              "assumptions": ["默认一日游"],
              "warnings": ["交通耗时证据不足"]
            }
            """
        )
        builder = ItineraryBuilder(llm=llm)

        result = builder.build(make_state())

        self.assertEqual(result["days"][0]["date_label"], "第 1 天")
        self.assertEqual(result["days"][0]["slots"][0]["title"], "象鼻山")
        self.assertEqual(result["days"][0]["slots"][0]["source_refs"], ["evidence_1"])
        self.assertEqual(result["assumptions"], ["默认一日游"])
        self.assertEqual(builder.last_metrics["itinerary_day_count"], 1)
        self.assertIn("结构化行程 JSON", llm.calls[0])

    def test_parse_failure_returns_warning_plan(self):
        builder = ItineraryBuilder(llm=FakeLLM("not json"))

        result = builder.build(make_state())

        self.assertEqual(result["days"], [])
        self.assertIn("结构化行程解析失败", result["warnings"][0])


if __name__ == "__main__":
    unittest.main()


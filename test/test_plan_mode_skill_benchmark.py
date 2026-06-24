import json
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from server.routers import query as query_router
from server.schemas import QueryRequest


DATASET = Path("evaluation/datasets/plan_mode_skill_qa.jsonl")


class FakeEngine:
    def __init__(self):
        self.calls = []

    def ask(self, question, route, report_mode="concise", plan_mode="auto", response_format="text"):
        self.calls.append(
            {
                "question": question,
                "route": route,
                "report_mode": report_mode,
                "plan_mode": plan_mode,
                "response_format": response_format,
            }
        )
        if plan_mode == "detailed_itinerary":
            answer = "上午游览核心景点，下午衔接交通，补充票价和预算。"
        elif plan_mode == "place_recommendations":
            answer = "推荐几个项目，说明亮点、适合人群和注意事项。"
        else:
            answer = "ok"
        return SimpleNamespace(route="agent", answer=answer, sources=[], metadata={})


def load_samples():
    samples = []
    with DATASET.open("r", encoding="utf-8") as file:
        for line in file:
            line = line.strip()
            if line:
                samples.append(json.loads(line))
    return samples


class PlanModeSkillBenchmarkTest(unittest.TestCase):
    def test_plan_mode_skill_api_benchmark(self):
        samples = load_samples()
        self.assertEqual(len(samples), 3)

        engine = FakeEngine()
        records = []

        with (
            patch.object(query_router, "check_llm_config", return_value=(True, None)),
            patch.object(query_router, "check_faiss_index", return_value=True),
            patch.object(query_router, "check_neo4j", return_value=True),
            patch.object(query_router, "append_trace", side_effect=records.append),
            patch.object(query_router, "get_query_engine", return_value=engine),
        ):
            for sample in samples:
                response = query_router._query_with_route(
                    QueryRequest(
                        question=sample["question"],
                        route=sample["route"],
                        plan_mode=sample["plan_mode"],
                    ),
                    forced_route=sample["route"],
                )

                self.assertEqual(response.route, sample["expected_route"], sample["id"])
                for keyword in sample["expected_answer_keywords"]:
                    self.assertIn(keyword, response.answer, sample["id"])

                if sample["expected_engine_call"]:
                    self.assertEqual(engine.calls[-1]["plan_mode"], sample["expected_engine_plan_mode"])
                    self.assertEqual(response.metadata.plan_mode, sample["expected_engine_plan_mode"])
                else:
                    self.assertEqual(len(engine.calls), 0)
                    self.assertTrue(response.metadata.engine_metadata["clarification_required"])

        self.assertEqual(len(records), len(samples))
        self.assertEqual(records[0]["actual_route"], "clarification")
        self.assertEqual(records[1]["plan_mode"], "detailed_itinerary")
        self.assertEqual(records[2]["plan_mode"], "place_recommendations")


if __name__ == "__main__":
    unittest.main()

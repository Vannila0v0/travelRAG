import unittest
from types import SimpleNamespace
from unittest.mock import patch

from agent_system.skills.travel_plan_clarifier import TravelPlanClarifierSkill
from server.routers import query as query_router
from server.schemas import QueryRequest


class TravelPlanClarifierSkillTest(unittest.TestCase):
    def test_broad_trip_plan_requires_clarification(self):
        skill = TravelPlanClarifierSkill()

        result = skill.match("我想去桂林玩三天，帮我安排一下")

        self.assertIsNotNone(result)
        self.assertIn("详细路线安排", result.answer)
        self.assertIn("景点/项目推荐", result.answer)
        self.assertTrue(result.metadata["clarification_required"])
        self.assertEqual(result.metadata["skill_spec"], "agent_system/skills/travel_plan_clarifier.md")
        self.assertEqual(
            result.metadata["options"][0]["next_request"]["plan_mode"],
            "detailed_itinerary",
        )

    def test_skill_uses_markdown_runtime_config(self):
        skill = TravelPlanClarifierSkill()

        self.assertEqual(skill.config["name"], "travel_plan_clarifier")
        self.assertIn("三日游", skill.config["trigger_terms"])
        self.assertIn("agent", skill.config["route_scope"])

    def test_explicit_detail_mode_does_not_require_clarification(self):
        skill = TravelPlanClarifierSkill()

        result = skill.match("帮我规划一天桂林市区详细路线，包含交通和票价")

        self.assertIsNone(result)

    def test_explicit_recommendation_mode_does_not_require_clarification(self):
        skill = TravelPlanClarifierSkill()

        result = skill.match("推荐几个桂林好玩的地方，不用路线")

        self.assertIsNone(result)

    def test_route_outside_skill_scope_does_not_require_clarification(self):
        skill = TravelPlanClarifierSkill()

        result = skill.match("想去桂林玩三天，帮我规划一下", route="vector")

        self.assertIsNone(result)

    def test_selected_plan_mode_does_not_require_clarification(self):
        skill = TravelPlanClarifierSkill()

        result = skill.match(
            "想去桂林玩三天，帮我规划一下",
            route="agent",
            plan_mode="detailed_itinerary",
        )

        self.assertIsNone(result)

    def test_query_entry_returns_clarification_without_engine_call(self):
        records = []

        with (
            patch.object(query_router, "uuid4", return_value=SimpleNamespace(hex="trace-clarify")),
            patch.object(query_router, "append_trace", side_effect=records.append),
            patch.object(query_router, "check_llm_config") as mock_llm,
            patch.object(query_router, "get_query_engine") as mock_engine,
        ):
            response = query_router._query_with_route(
                QueryRequest(question="想去桂林玩三天，帮我规划一下"),
                forced_route="agent",
            )

        self.assertEqual(response.route, "clarification")
        self.assertEqual(response.metadata.trace_id, "trace-clarify")
        self.assertEqual(response.metadata.actual_route, "clarification")
        self.assertTrue(response.metadata.engine_metadata["clarification_required"])
        self.assertEqual(records[0]["actual_route"], "clarification")
        self.assertEqual(records[0]["skill"], "travel_plan_clarifier")
        self.assertEqual(records[0]["skill_spec"], "agent_system/skills/travel_plan_clarifier.md")
        mock_llm.assert_not_called()
        mock_engine.assert_not_called()

    def test_query_entry_uses_selected_plan_mode_without_clarifying(self):
        records = []
        engine_calls = []

        class FakeEngine:
            def ask(self, question, route, report_mode="concise", plan_mode="auto", response_format="text"):
                engine_calls.append(
                    {
                        "question": question,
                        "route": route,
                        "report_mode": report_mode,
                        "plan_mode": plan_mode,
                        "response_format": response_format,
                    }
                )
                return SimpleNamespace(route="agent", answer="ok", sources=[], metadata={})

        with (
            patch.object(query_router, "uuid4", return_value=SimpleNamespace(hex="trace-selected-mode")),
            patch.object(query_router, "check_llm_config", return_value=(True, None)),
            patch.object(query_router, "check_faiss_index", return_value=True),
            patch.object(query_router, "check_neo4j", return_value=True),
            patch.object(query_router, "append_trace", side_effect=records.append),
            patch.object(query_router, "get_query_engine", return_value=FakeEngine()),
        ):
            response = query_router._query_with_route(
                QueryRequest(
                    question="想去桂林玩三天，帮我规划一下",
                    plan_mode="detailed_itinerary",
                ),
                forced_route="agent",
            )

        self.assertEqual(response.route, "agent")
        self.assertEqual(engine_calls[0]["plan_mode"], "detailed_itinerary")
        self.assertEqual(records[0]["actual_route"], "agent")
        self.assertEqual(records[0]["plan_mode"], "detailed_itinerary")


if __name__ == "__main__":
    unittest.main()

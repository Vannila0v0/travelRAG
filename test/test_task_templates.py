import unittest
from datetime import date, timedelta

from agent_system.planner.task_template_matcher import TaskTemplateMatcher


class TaskTemplateTests(unittest.TestCase):
    def test_direct_url_reading_template_uses_web_fetch_then_reflection(self):
        matcher = TaskTemplateMatcher()

        graph = matcher.match("请总结这个网页：https://example.com/guide")

        self.assertIsNotNone(graph)
        self.assertEqual([node.task_type for node in graph.nodes], ["web_fetch", "reflection"])
        self.assertIn("https://example.com/guide", graph.nodes[0].description)
        self.assertEqual(graph.nodes[1].depends_on, ["task_001"])

    def test_external_web_evidence_template_uses_search_select_fetch_reflection(self):
        matcher = TaskTemplateMatcher()

        graph = matcher.match("请查一下这个政策的最新公告，并依据网页来源说明")

        self.assertIsNotNone(graph)
        self.assertEqual([node.task_type for node in graph.nodes], ["web_search", "source_select", "web_fetch", "reflection"])
        self.assertEqual(graph.nodes[1].depends_on, ["task_001"])
        self.assertEqual(graph.nodes[2].depends_on, ["task_002"])
        self.assertEqual(graph.nodes[3].depends_on, ["task_001", "task_003"])

    def test_weather_aware_itinerary_template_uses_weather_before_reflection(self):
        matcher = TaskTemplateMatcher()
        start = (date.today() + timedelta(days=2)).isoformat()

        graph = matcher.match(f"请帮我规划{start}到桂林的三日游路线安排")

        self.assertIsNotNone(graph)
        self.assertEqual(
            [node.task_type for node in graph.nodes],
            ["weather_query", "global_search", "local_search", "map_route", "reflection"],
        )
        self.assertEqual(graph.nodes[0].parameters["query"], f"请帮我规划{start}到桂林的三日游路线安排")
        self.assertEqual(graph.nodes[3].parameters["query"], f"请帮我规划{start}到桂林的三日游路线安排")
        self.assertEqual(graph.nodes[4].depends_on, ["task_001", "task_002", "task_003", "task_004"])

    def test_multi_place_route_order_template_uses_map_route_then_reflection(self):
        matcher = TaskTemplateMatcher()

        graph = matcher.match("从桂林站出发，象鼻山、靖江王府、东西巷怎么走比较不绕路？")

        self.assertIsNotNone(graph)
        self.assertEqual([node.task_type for node in graph.nodes], ["map_route", "reflection"])
        self.assertEqual(graph.nodes[0].parameters["query"], "从桂林站出发，象鼻山、靖江王府、东西巷怎么走比较不绕路？")
        self.assertEqual(graph.nodes[1].depends_on, ["task_001"])

    def test_longsheng_day_plan_uses_global_seed_task(self):
        matcher = TaskTemplateMatcher()

        graph = matcher.match("帮我规划龙胜温泉一日游，从龙胜县城汽车总站出发，包含交通、门票优惠和游玩重点")

        self.assertIsNotNone(graph)
        self.assertEqual(graph.nodes[0].task_type, "global_search")
        self.assertIn("龙胜温泉", graph.nodes[0].entities)
        self.assertTrue(any(node.task_type == "local_search" for node in graph.nodes))

    def test_guilin_city_day_plan_uses_map_route_before_reflection(self):
        matcher = TaskTemplateMatcher()

        graph = matcher.match("帮我规划桂林市区一日游，包含交通、门票和预算")

        self.assertIsNotNone(graph)
        self.assertEqual([node.task_type for node in graph.nodes], ["global_search", "local_search", "local_search", "local_search", "map_route", "reflection"])
        self.assertEqual(graph.nodes[4].parameters["query"], "帮我规划桂林市区一日游，包含交通、门票和预算")
        self.assertEqual(graph.nodes[5].depends_on, ["task_002", "task_003", "task_004", "task_005"])


if __name__ == "__main__":
    unittest.main()

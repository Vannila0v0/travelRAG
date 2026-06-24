import unittest
from types import SimpleNamespace
from unittest.mock import patch

from agent_system.executor.tool_registry import TOOL_REGISTRY, TOOL_SPECS, format_tool_specs_for_planner
from agent_system.planner.task_decomposer import TaskDecomposer


class FakeLLM:
    def __init__(self):
        self.calls = []

    def invoke(self, prompt):
        self.calls.append(prompt)
        return SimpleNamespace(
            content=(
                '{"nodes":[{"task_id":"raw_1","task_type":"web_search",'
                '"description":"查询今天是否开放","priority":1,'
                '"depends_on":[],"entities":["两江四湖"],"status":"pending"}],'
                '"execution_mode":"sequential"}'
            )
        )


class ToolSpecsTest(unittest.TestCase):
    def test_tool_specs_cover_registered_tools(self):
        self.assertEqual(set(TOOL_REGISTRY), set(TOOL_SPECS))

    def test_tool_specs_include_plan_policy_metadata(self):
        for name, spec in TOOL_SPECS.items():
            self.assertEqual(spec.name, name)
            self.assertTrue(spec.plan_policy.importance)
            self.assertTrue(spec.plan_policy.merge_strategy)
        self.assertEqual(TOOL_SPECS["reflection"].plan_policy.importance, "critical")
        self.assertEqual(TOOL_SPECS["local_search"].plan_policy.merge_strategy, "same_tool_merge_entities")
        self.assertTrue(TOOL_SPECS["weather_query"].plan_policy.realtime_sensitive)

    def test_format_tool_specs_for_planner_includes_web_search_guidance(self):
        text = format_tool_specs_for_planner()

        self.assertIn("web_search", text)
        self.assertIn("web_fetch", text)
        self.assertIn("最新", text)
        self.assertIn("避免场景", text)

    def test_task_decomposer_injects_tool_specs_into_prompt(self):
        fake_llm = FakeLLM()

        with patch("agent_system.planner.task_decomposer.get_llm_model", return_value=fake_llm):
            decomposer = TaskDecomposer()
            graph = decomposer.decompose("今天两江四湖是否开放？")

        self.assertEqual(graph.nodes[0].task_type, "web_search")
        self.assertIn("web_search", fake_llm.calls[0])
        self.assertIn("实时网页信息", fake_llm.calls[0])
        self.assertIn("严格遵守工具说明", fake_llm.calls[0])


if __name__ == "__main__":
    unittest.main()

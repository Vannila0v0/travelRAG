import unittest
from types import SimpleNamespace
from unittest.mock import patch

from agent_system.executor.tool_registry import ReflectionTool, TOOL_REGISTRY


class FakeLLM:
    def __init__(self):
        self.calls = []

    def invoke(self, prompt):
        self.calls.append(prompt)
        return SimpleNamespace(content="综合校验结果")


class ReflectionToolTests(unittest.TestCase):
    def test_reflection_uses_previous_records_without_query_engine(self):
        llm = FakeLLM()
        tool = ReflectionTool(llm=llm)
        payload = {
            "description": "综合交通和门票信息",
            "previous_records": [
                {
                    "task_id": "task_001",
                    "route": "hybrid",
                    "inputs": {
                        "task_type": "local_search",
                        "description": "查询交通",
                    },
                    "output": "可以从龙胜县城汽车总站乘车前往温泉景区。",
                    "metadata": {"error": None},
                    "sources": [
                        {
                            "doc_id": "doc-1",
                            "chunk_id": "chunk-1",
                            "file_name": "tourism_dpo.md",
                            "source_path": "data/tourism_dpo.md",
                        }
                    ],
                }
            ],
        }

        with patch("agent_system.executor.tool_registry.get_query_engine", side_effect=AssertionError("should not query")):
            result = tool.structured_search(payload)

        self.assertTrue(result["success"])
        self.assertEqual(result["answer"], "综合校验结果")
        self.assertEqual(result["data"]["route"], "reflection")
        self.assertEqual(result["data"]["metadata"]["records_used"], 1)
        self.assertEqual(result["data"]["metadata"]["llm_calls"], 1)
        self.assertEqual(result["data"]["sources"][0]["chunk_id"], "chunk-1")
        self.assertEqual(len(llm.calls), 1)
        self.assertIn("可以从龙胜县城汽车总站乘车前往温泉景区", llm.calls[0])

    def test_registry_returns_reflection_tool(self):
        tool = TOOL_REGISTRY["reflection"]()

        self.assertIsInstance(tool, ReflectionTool)


if __name__ == "__main__":
    unittest.main()

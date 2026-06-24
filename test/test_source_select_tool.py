import json
import unittest
from types import SimpleNamespace

from agent_system.executor.tool_registry import SourceSelectTool, TOOL_REGISTRY


class FakeSelectLLM:
    def __init__(self, content: str):
        self.content = content
        self.calls: list[str] = []

    def invoke(self, prompt: str):
        self.calls.append(prompt)
        return SimpleNamespace(content=self.content)


class SourceSelectToolTest(unittest.TestCase):
    def test_registry_returns_source_select_tool(self):
        tool = TOOL_REGISTRY["source_select"]()

        self.assertIsInstance(tool, SourceSelectTool)

    def test_source_select_uses_llm_selected_index(self):
        llm = FakeSelectLLM(
            json.dumps(
                {
                    "selected_source_index": 2,
                    "confidence": 0.82,
                    "reason": "第二条是官方文档。",
                },
                ensure_ascii=False,
            )
        )
        previous_records = [
            {
                "sources": [
                    {
                        "title": "镜像站",
                        "url": "https://mirror.example.com/doc",
                        "source_type": "web",
                        "text": "镜像内容",
                    },
                    {
                        "title": "官方文档",
                        "url": "https://example.com/docs",
                        "source_type": "web",
                        "text": "官方说明",
                    },
                ]
            }
        ]

        result = SourceSelectTool(llm=llm).structured_search(
            {
                "query": "查官方文档",
                "description": "选择最合适的来源",
                "previous_records": previous_records,
            }
        )

        self.assertTrue(result["success"])
        self.assertEqual(result["data"]["route"], "source_select")
        self.assertEqual(result["data"]["metadata"]["selected_source_index"], 2)
        self.assertEqual(result["data"]["metadata"]["selected_url"], "https://example.com/docs")
        self.assertEqual(result["data"]["sources"][0]["url"], "https://example.com/docs")
        self.assertIn("官方文档", llm.calls[0])

    def test_source_select_returns_error_for_invalid_json(self):
        result = SourceSelectTool(llm=FakeSelectLLM("not json")).structured_search(
            {
                "query": "查官方文档",
                "previous_records": [
                    {"sources": [{"title": "官方文档", "url": "https://example.com/docs"}]}
                ],
            }
        )

        self.assertFalse(result["success"])
        self.assertIn("JSON parse failed", result["error"])
        self.assertEqual(result["data"]["metadata"]["selected_source_index"], None)


if __name__ == "__main__":
    unittest.main()

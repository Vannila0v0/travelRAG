import json
import os
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from agent_system.core.plan_spec import TaskNode
from agent_system.core.state import PlanExecuteState
from agent_system.executor.retrieval_executor import RetrievalExecutor
from agent_system.executor.tool_registry import TOOL_REGISTRY, WebSearchTool
from query_engine.router import QueryEngine


class WebSearchToolTest(unittest.TestCase):
    def tearDown(self):
        for key in [
            "WEB_SEARCH_PROVIDER",
            "WEB_SEARCH_API_KEY",
            "EXA_API_KEY",
            "WEB_SEARCH_MOCK_RESULTS",
            "WEB_SEARCH_MAX_RESULTS",
            "WEB_SEARCH_EXA_MCP_ENDPOINT",
            "WEB_SEARCH_EXA_TOOL_NAME",
        ]:
            os.environ.pop(key, None)

    def test_registry_returns_web_search_tool(self):
        tool = TOOL_REGISTRY["web_search"]()

        self.assertIsInstance(tool, WebSearchTool)

    def test_unconfigured_web_search_returns_clear_error(self):
        tool = WebSearchTool()

        result = tool.structured_search({"description": "桂林今天两江四湖是否开放"})

        self.assertFalse(result["success"])
        self.assertEqual(result["data"]["route"], "web_search")
        self.assertFalse(result["data"]["metadata"]["configured"])
        self.assertIn("尚未配置", result["answer"])

    def test_mock_web_search_returns_web_sources(self):
        os.environ["WEB_SEARCH_PROVIDER"] = "mock"
        os.environ["WEB_SEARCH_MOCK_RESULTS"] = (
            '[{"title":"两江四湖公告","url":"https://example.com/notice",'
            '"content":"今日开放，以官方公告为准。","score":0.9}]'
        )
        tool = WebSearchTool()

        result = tool.structured_search({"description": "桂林今天两江四湖是否开放"})

        self.assertTrue(result["success"])
        self.assertEqual(result["data"]["route"], "web_search")
        self.assertEqual(result["data"]["metadata"]["provider"], "mock")
        self.assertEqual(result["data"]["sources"][0]["url"], "https://example.com/notice")
        self.assertEqual(result["data"]["sources"][0]["source_type"], "web")
        self.assertIn("两江四湖公告", result["answer"])

    def test_exa_mcp_web_search_parses_sse_response(self):
        os.environ["WEB_SEARCH_PROVIDER"] = "exa_mcp"
        os.environ["WEB_SEARCH_API_KEY"] = "test-exa-key"
        sse_text = (
            "event: message\n"
            'data: {"jsonrpc":"2.0","id":"web_search_exa","result":{"content":[{"type":"text",'
            '"text":"{\\"results\\":[{\\"title\\":\\"Guilin notice\\",'
            '\\"url\\":\\"https://example.com/open\\",'
            '\\"content\\":\\"Open today\\",'
            '\\"published_date\\":\\"2026-06-17\\",\\"score\\":0.8}]}"}]}}\n\n'
        )

        class FakeResponse:
            headers = {"content-type": "text/event-stream"}
            text = sse_text

            def raise_for_status(self):
                return None

            def json(self):
                raise AssertionError("SSE response should be parsed from text")

        with patch("agent_system.executor.tool_registry.httpx.post", return_value=FakeResponse()) as post:
            result = WebSearchTool().structured_search({"description": "Guilin latest opening notice"})

        self.assertTrue(result["success"])
        self.assertEqual(result["data"]["metadata"]["provider"], "exa_mcp")
        self.assertEqual(result["data"]["metadata"]["tool_name"], "web_search_exa")
        self.assertEqual(result["data"]["metadata"]["endpoint"], "https://mcp.exa.ai/mcp")
        self.assertEqual(result["data"]["sources"][0]["url"], "https://example.com/open")
        self.assertEqual(result["data"]["sources"][0]["title"], "Guilin notice")
        self.assertEqual(result["data"]["sources"][0]["source_type"], "web")
        call_kwargs = post.call_args.kwargs
        self.assertEqual(call_kwargs["json"]["method"], "tools/call")
        self.assertEqual(call_kwargs["json"]["params"]["name"], "web_search_exa")
        self.assertEqual(call_kwargs["json"]["params"]["arguments"]["numResults"], 5)
        self.assertEqual(call_kwargs["headers"]["authorization"], "Bearer test-exa-key")

    def test_exa_mcp_public_endpoint_does_not_require_api_key(self):
        os.environ["WEB_SEARCH_PROVIDER"] = "exa_mcp"
        sse_text = (
            "event: message\n"
            'data: {"jsonrpc":"2.0","id":"web_search_exa","result":{"content":[{"type":"text",'
            '"text":"{\\"results\\":[{\\"title\\":\\"Public result\\",'
            '\\"url\\":\\"https://example.com/public\\",\\"content\\":\\"Public content\\"}]}"}]}}\n\n'
        )

        class FakeResponse:
            headers = {"content-type": "text/event-stream"}
            text = sse_text

            def raise_for_status(self):
                return None

        with patch("agent_system.executor.tool_registry.httpx.post", return_value=FakeResponse()) as post:
            result = WebSearchTool().structured_search({"description": "public mcp search"})

        self.assertTrue(result["success"])
        self.assertEqual(result["data"]["metadata"]["endpoint"], "https://mcp.exa.ai/mcp")
        self.assertEqual(result["data"]["sources"][0]["url"], "https://example.com/public")
        self.assertNotIn("authorization", post.call_args.kwargs["headers"])

    def test_exa_mcp_parses_plain_text_title_url_results(self):
        os.environ["WEB_SEARCH_PROVIDER"] = "exa_mcp"
        plain_results = (
            "Title: OpenAI Responses API\n"
            "URL: https://platform.openai.com/docs/api-reference/responses\n"
            "Published: N/A\n"
            "Author: N/A\n"
            "Highlights: Responses API reference page.\n\n"
            "Title: OpenAI Web Search\n"
            "URL: https://platform.openai.com/docs/guides/tools-web-search\n"
            "Published: 2026-06-19\n"
            "Author: N/A\n"
            "Highlights: Web search guide.\n"
        )
        sse_text = (
            "event: message\n"
            + "data: "
            + json.dumps(
                {
                    "jsonrpc": "2.0",
                    "id": "web_search_exa",
                    "result": {"content": [{"type": "text", "text": plain_results}]},
                }
            )
            + "\n\n"
        )

        class FakeResponse:
            headers = {"content-type": "text/event-stream"}
            text = sse_text

            def raise_for_status(self):
                return None

        with patch("agent_system.executor.tool_registry.httpx.post", return_value=FakeResponse()):
            result = WebSearchTool().structured_search({"description": "Responses API docs"})

        self.assertTrue(result["success"])
        self.assertEqual(result["data"]["metadata"]["result_count"], 2)
        self.assertEqual(result["data"]["sources"][0]["title"], "OpenAI Responses API")
        self.assertEqual(result["data"]["sources"][0]["url"], "https://platform.openai.com/docs/api-reference/responses")
        self.assertEqual(result["data"]["sources"][0]["text"], "Responses API reference page.")
        self.assertEqual(result["data"]["sources"][1]["published_at"], "2026-06-19")

    def test_retrieval_executor_can_run_web_search_task(self):
        os.environ["WEB_SEARCH_PROVIDER"] = "mock"
        os.environ["WEB_SEARCH_MOCK_RESULTS"] = (
            '[{"title":"开放公告","url":"https://example.com/open",'
            '"content":"今日开放。"}]'
        )
        task = TaskNode(
            task_id="task-web",
            task_type="web_search",
            description="查询桂林今天两江四湖是否开放",
        )

        record = RetrievalExecutor().execute_task(task, PlanExecuteState(session_id="s", input_query="q"))

        self.assertEqual(record.route, "web_search")
        self.assertEqual(record.sources[0]["url"], "https://example.com/open")
        self.assertEqual(record.tool_metadata["provider"], "mock")
        self.assertIsNone(record.metadata.error)

    def test_agent_source_conversion_preserves_web_url(self):
        state = PlanExecuteState(session_id="s", input_query="q")
        state.final_report = "answer"
        state.sources = [
            {
                "doc_id": "web:1",
                "chunk_id": "web:1",
                "title": "公告",
                "url": "https://example.com/open",
                "source_type": "web",
                "text": "今日开放。",
            }
        ]

        class FakeOrchestrator:
            def run(self, query, report_mode="concise", plan_mode="auto"):
                return state

        with patch("agent_system.orchestrator.MultiAgentOrchestrator", return_value=FakeOrchestrator()):
            result = QueryEngine(llm=SimpleNamespace()).ask("q", route="agent")

        self.assertEqual(result.sources[0].url, "https://example.com/open")
        self.assertEqual(result.sources[0].source_type, "web")
        self.assertEqual(result.sources[0].title, "公告")


if __name__ == "__main__":
    unittest.main()

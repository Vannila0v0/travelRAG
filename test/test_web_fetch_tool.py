import json
import os
import unittest
from unittest.mock import patch

from agent_system.core.plan_spec import TaskNode
from agent_system.core.state import PlanExecuteState
from agent_system.executor.retrieval_executor import RetrievalExecutor
from agent_system.executor.tool_registry import TOOL_REGISTRY, WebFetchTool


class FakeStreamResponse:
    def __init__(
        self,
        body: bytes,
        *,
        url: str = "https://example.com/page",
        status_code: int = 200,
        headers: dict[str, str] | None = None,
        encoding: str = "utf-8",
    ):
        self._body = body
        self.url = url
        self.status_code = status_code
        self.headers = headers or {"content-type": "text/html; charset=utf-8"}
        self.encoding = encoding

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False

    def raise_for_status(self):
        if self.status_code >= 400:
            raise RuntimeError(f"status={self.status_code}")

    def iter_bytes(self):
        yield self._body


class WebFetchToolTest(unittest.TestCase):
    def tearDown(self):
        for key in [
            "WEB_FETCH_TIMEOUT_SECONDS",
            "WEB_FETCH_MAX_BYTES",
            "WEB_FETCH_MAX_CHARS",
            "WEB_FETCH_MAX_REDIRECTS",
            "WEB_FETCH_USER_AGENT",
        ]:
            os.environ.pop(key, None)

    def test_registry_returns_web_fetch_tool(self):
        tool = TOOL_REGISTRY["web_fetch"]()

        self.assertIsInstance(tool, WebFetchTool)

    def test_web_fetch_converts_html_to_json_text(self):
        html = (
            b"<html><head><title>Notice</title><style>.x{}</style></head>"
            b"<body><h1>Opening Notice</h1><p>Open today.</p>"
            b"<script>alert(1)</script><a href='https://example.com/detail'>Detail</a></body></html>"
        )

        with patch(
            "agent_system.executor.tool_registry.httpx.stream",
            return_value=FakeStreamResponse(html),
        ) as stream:
            result = WebFetchTool().structured_search({"url": "https://example.com/page"})

        self.assertTrue(result["success"])
        payload = json.loads(result["answer"])
        self.assertEqual(payload["url"], "https://example.com/page")
        self.assertEqual(payload["title"], "Notice")
        self.assertIn("Opening Notice", payload["text"])
        self.assertIn("Open today.", payload["text"])
        self.assertNotIn("alert", payload["text"])
        self.assertEqual(result["data"]["route"], "web_fetch")
        self.assertEqual(result["data"]["sources"][0]["url"], "https://example.com/page")
        self.assertEqual(result["data"]["sources"][0]["source_type"], "web")
        self.assertEqual(stream.call_args.args[0], "GET")

    def test_web_fetch_blocks_private_ip_url(self):
        result = WebFetchTool().structured_search({"url": "http://127.0.0.1:8000/admin"})

        self.assertFalse(result["success"])
        self.assertEqual(result["data"]["metadata"]["reason"], "unsafe_url")
        self.assertIn("不允许访问", result["error"])

    def test_web_fetch_requires_url_or_source_index(self):
        result = WebFetchTool().structured_search({"description": "读取前序网页正文"})

        self.assertFalse(result["success"])
        self.assertEqual(result["data"]["metadata"]["reason"], "missing_url")
        self.assertIn("url 或 source_index", result["error"])

    def test_web_fetch_uses_source_index_from_previous_records(self):
        response = FakeStreamResponse(
            b"<html><head><title>Second</title></head><body><p>Second page.</p></body></html>",
            url="https://example.com/second",
        )
        previous_records = [
            {
                "sources": [
                    {"url": "https://example.com/first", "title": "First"},
                    {"url": "https://example.com/second", "title": "Second"},
                ]
            }
        ]

        with patch("agent_system.executor.tool_registry.httpx.stream", return_value=response) as stream:
            result = WebFetchTool().structured_search(
                {
                    "source_index": 2,
                    "previous_records": previous_records,
                }
            )

        self.assertTrue(result["success"])
        self.assertEqual(stream.call_args.args[1], "https://example.com/second")
        self.assertEqual(result["data"]["sources"][0]["url"], "https://example.com/second")

    def test_web_fetch_rejects_out_of_range_source_index(self):
        result = WebFetchTool().structured_search(
            {
                "source_index": 2,
                "previous_records": [{"sources": [{"url": "https://example.com/only"}]}],
            }
        )

        self.assertFalse(result["success"])
        self.assertEqual(result["data"]["metadata"]["reason"], "source_index_out_of_range")
        self.assertIn("越界", result["error"])

    def test_web_fetch_rejects_oversized_response(self):
        os.environ["WEB_FETCH_MAX_BYTES"] = "4"
        response = FakeStreamResponse(
            b"too large",
            headers={"content-type": "text/plain", "content-length": "9"},
        )

        with patch("agent_system.executor.tool_registry.httpx.stream", return_value=response):
            result = WebFetchTool().structured_search({"url": "https://example.com/big.txt"})

        self.assertFalse(result["success"])
        self.assertEqual(result["data"]["metadata"]["reason"], "fetch_failed")
        self.assertIn("too large", result["error"])

    def test_retrieval_executor_can_run_web_fetch_task(self):
        response = FakeStreamResponse(
            b'{"status":"open"}',
            headers={"content-type": "application/json"},
        )
        task = TaskNode(
            task_id="task-fetch",
            task_type="web_fetch",
            description="读取 https://example.com/api",
            parameters={"url": "https://example.com/api"},
        )

        with patch("agent_system.executor.tool_registry.httpx.stream", return_value=response):
            record = RetrievalExecutor().execute_task(task, PlanExecuteState(session_id="s", input_query="q"))

        self.assertEqual(record.route, "web_fetch")
        self.assertEqual(record.sources[0]["url"], "https://example.com/page")
        self.assertIsNone(record.metadata.error)


if __name__ == "__main__":
    unittest.main()

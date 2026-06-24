import json
import os
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from query_engine.router import QueryEngine


class RoutingFakeLLM:
    def __init__(self):
        self.calls: list[str] = []

    def invoke(self, prompt: str):
        self.calls.append(prompt)
        if "请严格按照以下 JSON 格式输出 TaskGraph" in prompt:
            return SimpleNamespace(
                content=json.dumps(
                    {
                        "nodes": [
                            {
                                "task_id": "fallback_1",
                                "task_type": "local_search",
                                "description": "fallback should be replaced by task template",
                                "priority": 1,
                                "depends_on": [],
                                "entities": [],
                                "parameters": {},
                                "status": "pending",
                            }
                        ],
                        "execution_mode": "sequential",
                    },
                    ensure_ascii=False,
                )
            )
        if "来源选择助手" in prompt:
            return SimpleNamespace(
                content=json.dumps(
                    {
                        "selected_source_index": 2,
                        "confidence": 0.9,
                        "reason": "第二条来源是更合适的网页正文。",
                    },
                    ensure_ascii=False,
                )
            )
        if "当前反思任务" in prompt:
            return SimpleNamespace(content="网页证据充分，页面正文显示 Open today。")
        return SimpleNamespace(content="最终回答：根据网页证据，Open today。")


class FakeStreamResponse:
    def __init__(
        self,
        body: bytes,
        *,
        url: str,
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


def fake_stream(method, url, **kwargs):
    html = (
        b"<html><head><title>Official Notice</title></head>"
        b"<body><h1>Official Notice</h1><p>Open today.</p></body></html>"
    )
    return FakeStreamResponse(html, url=url)


class QueryEngineWebToolsTest(unittest.TestCase):
    def tearDown(self):
        for key in [
            "WEB_SEARCH_PROVIDER",
            "WEB_SEARCH_MOCK_RESULTS",
            "WEB_FETCH_MAX_BYTES",
            "WEB_FETCH_MAX_CHARS",
            "AGENT_MAX_WORKERS",
        ]:
            os.environ.pop(key, None)

    def _run_agent_query(self, question: str):
        os.environ["AGENT_MAX_WORKERS"] = "1"
        os.environ["WEB_SEARCH_PROVIDER"] = "mock"
        os.environ["WEB_SEARCH_MOCK_RESULTS"] = json.dumps(
            [
                {
                    "title": "两江四湖官方公告",
                    "url": "https://example.com/notice",
                    "content": "今日开放，以官方公告为准。",
                    "score": 0.9,
                },
                {
                    "title": "两江四湖官方正文",
                    "url": "https://example.com/official",
                    "content": "Open today.",
                    "score": 0.95,
                }
            ],
            ensure_ascii=False,
        )
        fake_llm = RoutingFakeLLM()
        with (
            patch("agent_system.planner.task_decomposer.get_llm_model", return_value=fake_llm),
            patch("agent_system.executor.tool_registry.get_llm_model", return_value=fake_llm),
            patch("agent_system.reporter.base_reporter.get_llm_model", return_value=fake_llm),
            patch("agent_system.executor.tool_registry.httpx.stream", side_effect=fake_stream),
        ):
            result = QueryEngine(llm=SimpleNamespace()).ask(question, route="agent")
        return result

    def test_agent_direct_url_template_runs_end_to_end(self):
        result = self._run_agent_query("请读取并总结这个网页：https://example.com/guide")

        self.assertEqual(result.route, "agent")
        self.assertIn("Open today", result.answer)
        self.assertEqual(
            [task["task_type"] for task in result.metadata["tasks"]],
            ["web_fetch", "reflection"],
        )
        self.assertEqual(
            [task["route"] for task in result.metadata["agent_trace"]["tasks"]],
            ["web_fetch", "reflection"],
        )
        self.assertEqual(result.sources[0].url, "https://example.com/guide")
        self.assertEqual(result.sources[0].source_type, "web")

    def test_agent_external_web_evidence_template_runs_search_fetch_reflection(self):
        result = self._run_agent_query("桂林两江四湖今天是否开放？请查最新信息")

        self.assertEqual(result.route, "agent")
        self.assertIn("Open today", result.answer)
        tasks = result.metadata["tasks"]
        self.assertEqual(
            [task["task_type"] for task in tasks],
            ["web_search", "source_select", "web_fetch", "reflection"],
        )
        self.assertEqual(
            [task["route"] for task in result.metadata["agent_trace"]["tasks"]],
            ["web_search", "source_select", "web_fetch", "reflection"],
        )
        urls = {source.url for source in result.sources}
        self.assertIn("https://example.com/official", urls)


if __name__ == "__main__":
    unittest.main()

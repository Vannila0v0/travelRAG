import argparse
import io
import json
import os
import unittest
from contextlib import redirect_stdout
from unittest.mock import patch

import manage
from agent_system.core.execution_record import ExecutionMetadata, ExecutionRecord
from agent_system.core.plan_spec import PlanSpec, TaskGraph, TaskNode
from agent_system.core.state import PlanExecuteState


class FakeOrchestrator:
    def run(self, query, report_mode="concise", plan_mode="auto"):
        state = PlanExecuteState(session_id="smoke", input_query=query, plan_mode=plan_mode)
        state.plan = PlanSpec(
            original_query=query,
            task_graph=TaskGraph(
                execution_mode="sequential",
                nodes=[
                    TaskNode(
                        task_id="task_001",
                        task_type="web_search",
                        description="搜索外部网页来源",
                    ),
                    TaskNode(
                        task_id="task_002",
                        task_type="source_select",
                        description="选择最合适的网页来源",
                        depends_on=["task_001"],
                    ),
                    TaskNode(
                        task_id="task_003",
                        task_type="web_fetch",
                        description="读取第 1 条网页来源正文",
                        depends_on=["task_002"],
                    ),
                ],
            ),
        )
        state.execution_records = [
            ExecutionRecord(
                task_id="task_001",
                worker_type="retrieval_executor",
                inputs={"task_type": "web_search", "description": "搜索外部网页来源"},
                output="搜索结果",
                route="web_search",
                sources=[
                    {
                        "title": "公告",
                        "url": "https://example.com/notice",
                        "source_type": "web",
                        "text": "摘要",
                    }
                ],
                tool_metadata={"provider": "mock", "result_count": 1},
                metadata=ExecutionMetadata(worker_type="retrieval_executor", latency_seconds=0.01),
            ),
            ExecutionRecord(
                task_id="task_002",
                worker_type="retrieval_executor",
                inputs={"task_type": "source_select", "description": "选择最合适的网页来源"},
                output='{"selected_source_index":2}',
                route="source_select",
                sources=[
                    {
                        "title": "公告",
                        "url": "https://example.com/notice",
                        "source_type": "web",
                        "text": "摘要",
                    }
                ],
                tool_metadata={"selected_source_index": 1, "confidence": 0.9},
                metadata=ExecutionMetadata(worker_type="retrieval_executor", latency_seconds=0.01),
            ),
            ExecutionRecord(
                task_id="task_003",
                worker_type="retrieval_executor",
                inputs={"task_type": "web_fetch", "description": "读取第 1 条网页来源正文", "source_index": 1},
                output='{"text":"Open today"}',
                route="web_fetch",
                sources=[
                    {
                        "title": "公告",
                        "url": "https://example.com/notice",
                        "source_type": "web",
                        "text": "Open today",
                    }
                ],
                tool_metadata={"content_type": "text/html", "bytes_read": 42, "truncated": False},
                metadata=ExecutionMetadata(worker_type="retrieval_executor", latency_seconds=0.02),
            ),
        ]
        state.sources = state.execution_records[1].sources
        state.agent_trace = {"task_count": 2}
        state.final_report = "最终回答：Open today"
        return state


class ManageSmokeWebToolsTest(unittest.TestCase):
    def tearDown(self):
        for key in ["WEB_SEARCH_PROVIDER", "WEB_SEARCH_API_KEY", "EXA_API_KEY"]:
            os.environ.pop(key, None)

    def test_require_exa_fails_before_running_with_non_exa_provider(self):
        os.environ["WEB_SEARCH_PROVIDER"] = "mock"
        args = argparse.Namespace(
            query="q",
            report_mode="concise",
            plan_mode="auto",
            require_exa=True,
            json=False,
            show_outputs=False,
        )
        stdout = io.StringIO()

        with redirect_stdout(stdout):
            code = manage.smoke_web_tools_command(args)

        self.assertEqual(code, 2)
        self.assertIn("WEB_SEARCH_PROVIDER is not exa_mcp/exa", stdout.getvalue())

    def test_smoke_web_tools_json_outputs_records(self):
        os.environ["WEB_SEARCH_PROVIDER"] = "mock"
        args = argparse.Namespace(
            query="q",
            report_mode="concise",
            plan_mode="auto",
            require_exa=False,
            json=True,
            show_outputs=False,
        )
        stdout = io.StringIO()

        with (
            patch("agent_system.orchestrator.MultiAgentOrchestrator", return_value=FakeOrchestrator()),
            redirect_stdout(stdout),
        ):
            code = manage.smoke_web_tools_command(args)

        self.assertEqual(code, 0)
        payload = json.loads(stdout.getvalue())
        self.assertEqual(payload["execution_records"][0]["route"], "web_search")
        self.assertEqual(payload["execution_records"][1]["route"], "source_select")
        self.assertEqual(payload["execution_records"][2]["route"], "web_fetch")
        self.assertEqual(payload["sources"][0]["url"], "https://example.com/notice")
        self.assertIn("Open today", payload["final_answer"])


if __name__ == "__main__":
    unittest.main()

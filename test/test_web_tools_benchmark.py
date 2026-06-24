import json
import os
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from agent_system.core.plan_spec import PlanSpec
from agent_system.core.state import PlanExecuteState
from agent_system.executor.worker_coordinator import WorkerCoordinator
from agent_system.planner.task_decomposer import TaskDecomposer


DATASET = Path("evaluation/datasets/web_tools_qa.jsonl")


class FakeLLM:
    def __init__(self, content: str):
        self.content = content
        self.calls: list[str] = []

    def invoke(self, prompt: str):
        self.calls.append(prompt)
        return SimpleNamespace(content=self.content)


class FakeReflectionLLM:
    def invoke(self, prompt: str):
        if "来源选择助手" in prompt:
            return SimpleNamespace(
                content=json.dumps(
                    {
                        "selected_source_index": 2,
                        "confidence": 0.88,
                        "reason": "第二条来源更匹配官方公告正文。",
                    },
                    ensure_ascii=False,
                )
            )
        return SimpleNamespace(content="网页证据充分，Open today。")


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


def load_samples() -> list[dict]:
    samples = []
    with DATASET.open("r", encoding="utf-8") as file:
        for line in file:
            line = line.strip()
            if line:
                samples.append(json.loads(line))
    return samples


def fake_stream(method, url, **kwargs):
    html = (
        b"<html><head><title>Official Notice</title></head>"
        b"<body><h1>Official Notice</h1><p>Open today.</p></body></html>"
    )
    return FakeStreamResponse(html, url=url)


class WebToolsBenchmarkTest(unittest.TestCase):
    def tearDown(self):
        for key in [
            "WEB_SEARCH_PROVIDER",
            "WEB_SEARCH_MOCK_RESULTS",
            "WEB_FETCH_MAX_BYTES",
            "WEB_FETCH_MAX_CHARS",
        ]:
            os.environ.pop(key, None)

    def test_web_tools_planner_and_executor_benchmark(self):
        samples = load_samples()
        self.assertEqual(len(samples), 3)

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

        for sample in samples:
            fake_llm = FakeLLM(json.dumps(sample["planner_output"], ensure_ascii=False))
            with patch("agent_system.planner.task_decomposer.get_llm_model", return_value=fake_llm):
                graph = TaskDecomposer().decompose(sample["question"])

            task_types = [node.task_type for node in graph.nodes]
            self.assertEqual(task_types, sample["expected_task_types"], sample["id"])
            self.assertIn("web_search", fake_llm.calls[0], sample["id"])
            self.assertIn("web_fetch", fake_llm.calls[0], sample["id"])

            state = PlanExecuteState(session_id=sample["id"], input_query=sample["question"])
            state.plan = PlanSpec(original_query=sample["question"], task_graph=graph)
            with (
                patch("agent_system.executor.tool_registry.httpx.stream", side_effect=fake_stream),
                patch("agent_system.executor.tool_registry.get_llm_model", return_value=FakeReflectionLLM()),
            ):
                WorkerCoordinator(max_workers=1).run(state)

            routes = [record.route for record in state.execution_records]
            self.assertEqual(routes, sample["expected_routes"], sample["id"])
            self.assertFalse(
                [record.metadata.error for record in state.execution_records if record.metadata.error],
                sample["id"],
            )
            combined_output = "\n".join(str(record.output) for record in state.execution_records)
            for keyword in sample["expected_output_keywords"]:
                self.assertIn(keyword, combined_output, sample["id"])

        chained = samples[2]
        fake_llm = FakeLLM(json.dumps(chained["planner_output"], ensure_ascii=False))
        with patch("agent_system.planner.task_decomposer.get_llm_model", return_value=fake_llm):
            graph = TaskDecomposer().decompose(chained["question"])
        state = PlanExecuteState(session_id=chained["id"], input_query=chained["question"])
        state.plan = PlanSpec(original_query=chained["question"], task_graph=graph)
        with (
            patch("agent_system.executor.tool_registry.httpx.stream", side_effect=fake_stream),
            patch("agent_system.executor.tool_registry.get_llm_model", return_value=FakeReflectionLLM()),
        ):
            WorkerCoordinator(max_workers=1).run(state)

        fetch_record = next(record for record in state.execution_records if record.route == "web_fetch")
        self.assertNotIn("url", fetch_record.inputs)
        self.assertEqual(fetch_record.inputs["source_index"], 2)
        self.assertEqual(fetch_record.sources[0]["url"], "https://example.com/official")
        self.assertTrue(fetch_record.inputs["previous_records"])
        select_record = next(record for record in state.execution_records if record.route == "source_select")
        self.assertEqual(select_record.tool_metadata["selected_source_index"], 2)


if __name__ == "__main__":
    unittest.main()

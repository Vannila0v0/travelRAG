import json
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from server.routers import query as query_router
from server.schemas import QueryRequest


class FakeEngine:
    def __init__(self, result):
        self.result = result
        self.calls = []

    def ask(self, question, route, report_mode="concise", plan_mode="auto", response_format="text"):
        self.calls.append(
            {
                "question": question,
                "route": route,
                "report_mode": report_mode,
                "plan_mode": plan_mode,
                "response_format": response_format,
            }
        )
        return self.result


def make_result(route="vector", sources=None):
    return SimpleNamespace(
        route=route,
        answer="ok",
        sources=sources or [],
        metadata={"engine": "fake"},
        structured_output=None,
    )


class TraceLoggingTest(unittest.TestCase):
    def test_success_response_returns_and_writes_trace_id(self):
        records = []
        result = make_result(
            sources=[
                SimpleNamespace(
                    doc_id="doc-1",
                    chunk_id="chunk-1",
                    source_path="data/demo.md",
                    file_name="demo.md",
                    chunk_index=0,
                    page=None,
                    section=None,
                    text="source text",
                    score=0.9,
                )
            ]
        )

        with (
            patch.object(query_router, "uuid4", return_value=SimpleNamespace(hex="trace-success")),
            patch.object(query_router, "check_llm_config", return_value=(True, None)),
            patch.object(query_router, "check_faiss_index", return_value=True),
            patch.object(query_router, "append_trace", side_effect=records.append),
            patch.object(query_router, "get_query_engine", return_value=FakeEngine(result)),
        ):
            response = query_router._query_with_route(
                QueryRequest(question="两江四湖成人票多少钱？", route="vector")
            )

        self.assertEqual(response.metadata.trace_id, "trace-success")
        self.assertEqual(records[0]["trace_id"], "trace-success")
        self.assertTrue(records[0]["success"])
        self.assertEqual(records[0]["requested_route"], "vector")
        self.assertEqual(records[0]["actual_route"], "vector")
        self.assertEqual(records[0]["source_count"], 1)
        self.assertFalse(records[0]["degraded"])

    def test_preflight_failure_returns_and_writes_trace_id(self):
        records = []

        with (
            patch.object(query_router, "uuid4", return_value=SimpleNamespace(hex="trace-error")),
            patch.object(query_router, "check_llm_config", return_value=(False, "missing key")),
            patch.object(query_router, "append_trace", side_effect=records.append),
        ):
            response = query_router._query_with_route(
                QueryRequest(question="测试问题", route="vector")
            )

        payload = json.loads(response.body.decode("utf-8"))
        self.assertEqual(payload["metadata"]["trace_id"], "trace-error")
        self.assertEqual(payload["error"]["code"], "LLM_CONFIG_MISSING")
        self.assertEqual(records[0]["trace_id"], "trace-error")
        self.assertFalse(records[0]["success"])
        self.assertEqual(records[0]["error_code"], "LLM_CONFIG_MISSING")
        self.assertEqual(records[0]["source_count"], 0)

    def test_degraded_success_writes_degradation_fields(self):
        records = []
        result = make_result(route="vector")

        with (
            patch.object(query_router, "uuid4", return_value=SimpleNamespace(hex="trace-degraded")),
            patch.object(query_router, "check_llm_config", return_value=(True, None)),
            patch.object(query_router, "check_faiss_index", return_value=True),
            patch.object(query_router, "check_neo4j", side_effect=RuntimeError("neo4j down")),
            patch.object(query_router, "append_trace", side_effect=records.append),
            patch.object(query_router, "get_query_engine", return_value=FakeEngine(result)),
        ):
            response = query_router._query_with_route(
                QueryRequest(question="测试降级", route="hybrid", allow_degraded=True)
            )

        self.assertEqual(response.metadata.trace_id, "trace-degraded")
        self.assertTrue(response.metadata.degraded)
        self.assertEqual(response.metadata.degraded_from, "hybrid")
        self.assertEqual(response.metadata.degraded_to, "vector")
        self.assertTrue(records[0]["degraded"])
        self.assertEqual(records[0]["degraded_from"], "hybrid")
        self.assertEqual(records[0]["degraded_to"], "vector")
        self.assertEqual(records[0]["degradation_reason"], "NEO4J_UNAVAILABLE")

    def test_agent_success_writes_agent_trace(self):
        records = []
        agent_trace = {
            "planner_latency_ms": 10,
            "execution_latency_ms": 20,
            "reporter_latency_ms": 30,
            "task_count": 1,
            "tasks": [
                {
                    "task_id": "task-1",
                    "task_type": "local_search",
                    "status": "completed",
                    "latency_ms": 12,
                    "tool_name": "local_search",
                    "source_count": 2,
                }
            ],
            "cache_hits": 0,
            "cache_misses": 1,
        }
        result = make_result(route="agent")
        result.metadata = {"agent_trace": agent_trace}

        with (
            patch.object(query_router, "uuid4", return_value=SimpleNamespace(hex="trace-agent")),
            patch.object(query_router, "check_llm_config", return_value=(True, None)),
            patch.object(query_router, "check_faiss_index", return_value=True),
            patch.object(query_router, "check_neo4j", return_value=True),
            patch.object(query_router, "append_trace", side_effect=records.append),
            patch.object(query_router, "get_query_engine", return_value=FakeEngine(result)),
        ):
            response = query_router._query_with_route(
                QueryRequest(question="Agent 测试", route="agent")
            )

        self.assertEqual(response.metadata.trace_id, "trace-agent")
        self.assertEqual(response.metadata.engine_metadata["agent_trace"], agent_trace)
        self.assertEqual(records[0]["trace_id"], "trace-agent")
        self.assertEqual(records[0]["agent_trace"], agent_trace)

    def test_agent_web_tools_response_and_trace_preserve_sources(self):
        records = []
        agent_trace = {
            "task_count": 3,
            "tasks": [
                {"task_id": "task_001", "task_type": "web_search", "route": "web_search"},
                {"task_id": "task_002", "task_type": "source_select", "route": "source_select"},
                {"task_id": "task_003", "task_type": "web_fetch", "route": "web_fetch"},
                {"task_id": "task_004", "task_type": "reflection", "route": "reflection"},
            ],
        }
        result = make_result(
            route="agent",
            sources=[
                SimpleNamespace(
                    doc_id="web:1",
                    chunk_id="web:1",
                    source_path="https://example.com/notice",
                    file_name="两江四湖官方公告",
                    title="两江四湖官方公告",
                    url="https://example.com/notice",
                    published_at=None,
                    source_type="web",
                    chunk_index=None,
                    page=None,
                    section=None,
                    text="Open today.",
                    score=0.9,
                )
            ],
        )
        result.metadata = {"agent_trace": agent_trace}
        engine = FakeEngine(result)

        with (
            patch.object(query_router, "uuid4", return_value=SimpleNamespace(hex="trace-agent-web")),
            patch.object(query_router, "check_llm_config", return_value=(True, None)),
            patch.object(query_router, "check_faiss_index", return_value=True),
            patch.object(query_router, "check_neo4j", return_value=True),
            patch.object(query_router, "append_trace", side_effect=records.append),
            patch.object(query_router, "get_query_engine", return_value=engine),
        ):
            response = query_router._query_with_route(
                QueryRequest(question="桂林两江四湖今天是否开放？请查最新信息", route="agent"),
                forced_route="agent",
            )

        self.assertEqual(response.route, "agent")
        self.assertEqual(response.sources[0].url, "https://example.com/notice")
        self.assertEqual(response.sources[0].source_type, "web")
        self.assertEqual(
            [task["route"] for task in response.metadata.engine_metadata["agent_trace"]["tasks"]],
            ["web_search", "source_select", "web_fetch", "reflection"],
        )
        self.assertEqual(records[0]["trace_id"], "trace-agent-web")
        self.assertEqual(records[0]["agent_trace"], agent_trace)
        self.assertEqual(records[0]["source_count"], 1)

    def test_agent_query_passes_report_and_plan_mode(self):
        records = []
        result = make_result(route="agent")
        engine = FakeEngine(result)

        with (
            patch.object(query_router, "uuid4", return_value=SimpleNamespace(hex="trace-agent-full")),
            patch.object(query_router, "check_llm_config", return_value=(True, None)),
            patch.object(query_router, "check_faiss_index", return_value=True),
            patch.object(query_router, "check_neo4j", return_value=True),
            patch.object(query_router, "append_trace", side_effect=records.append),
            patch.object(query_router, "get_query_engine", return_value=engine),
        ):
            query_router._query_with_route(
                QueryRequest(
                    question="Agent full 测试",
                    route="agent",
                    report_mode="full",
                    plan_mode="detailed_itinerary",
                ),
                forced_route="agent",
            )

        self.assertEqual(engine.calls[0]["route"], "agent")
        self.assertEqual(engine.calls[0]["report_mode"], "full")
        self.assertEqual(engine.calls[0]["plan_mode"], "detailed_itinerary")

    def test_agent_query_passes_response_format_and_returns_structured_output(self):
        records = []
        result = make_result(route="agent")
        result.structured_output = {
            "days": [{"date_label": "第 1 天", "slots": [{"title": "象鼻山"}]}],
            "total_budget": "证据不足",
            "assumptions": [],
            "warnings": [],
        }
        result.metadata = {
            "itinerary_validation": {
                "valid": True,
                "issues": [],
                "stats": {"day_count": 1, "slot_count": 1},
            }
        }
        engine = FakeEngine(result)

        with (
            patch.object(query_router, "uuid4", return_value=SimpleNamespace(hex="trace-agent-itinerary")),
            patch.object(query_router, "check_llm_config", return_value=(True, None)),
            patch.object(query_router, "check_faiss_index", return_value=True),
            patch.object(query_router, "check_neo4j", return_value=True),
            patch.object(query_router, "append_trace", side_effect=records.append),
            patch.object(query_router, "get_query_engine", return_value=engine),
        ):
            response = query_router._query_with_route(
                QueryRequest(
                    question="Agent itinerary 测试",
                    route="agent",
                    plan_mode="detailed_itinerary",
                    response_format="itinerary",
                ),
                forced_route="agent",
            )

        self.assertEqual(engine.calls[0]["response_format"], "itinerary")
        self.assertEqual(response.metadata.response_format, "itinerary")
        self.assertEqual(response.structured_output.days[0].date_label, "第 1 天")
        self.assertEqual(response.structured_output.days[0].slots[0].title, "象鼻山")
        self.assertEqual(records[0]["response_format"], "itinerary")
        self.assertEqual(records[0]["structured_output"]["days"][0]["date_label"], "第 1 天")
        self.assertTrue(records[0]["itinerary_validation"]["valid"])


if __name__ == "__main__":
    unittest.main()

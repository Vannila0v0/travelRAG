import unittest
from unittest.mock import patch

from fastapi.testclient import TestClient

from server.app import app
from server.routers import dashboard as dashboard_router


class DashboardApiTest(unittest.TestCase):
    def setUp(self):
        self.client = TestClient(app)

    def test_summary_returns_trace_statistics(self):
        summary = {
            "total": 3,
            "success": 2,
            "errors": 1,
            "degraded": 1,
            "avg_latency_ms": 123.45,
            "routes": {"vector": 2, "hybrid": 1},
            "error_codes": {"QUERY_FAILED": 1},
        }

        with patch.object(dashboard_router, "summarize_traces", return_value=summary) as mock_summary:
            response = self.client.get("/dashboard/summary?limit=20")

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json(), summary)
        mock_summary.assert_called_once_with(limit=20)

    def test_dashboard_page_returns_html(self):
        summary = {
            "total": 2,
            "success": 1,
            "errors": 1,
            "degraded": 1,
            "avg_latency_ms": 88,
            "routes": {"vector": 1, "hybrid": 1},
            "error_codes": {"QUERY_FAILED": 1},
        }
        traces = [
            {
                "trace_id": "trace-page",
                "timestamp": "2026-06-14T12:00:00",
                "question": "页面测试",
                "requested_route": "hybrid",
                "actual_route": "vector",
                "latency_ms": 88,
                "success": True,
                "error_code": None,
                "source_count": 2,
                "degraded": True,
                "degraded_from": "hybrid",
                "degraded_to": "vector",
                "degradation_reason": "NEO4J_UNAVAILABLE",
                "structured_output": {
                    "days": [
                        {
                            "date_label": "第 1 天",
                            "slots": [
                                {
                                    "start_time": "09:00",
                                    "end_time": "11:00",
                                    "title": "象鼻山",
                                    "location": "桂林市区",
                                    "activity": "游览象鼻山",
                                    "transport_to_next": "打车前往两江四湖",
                                    "ticket_info": "以证据为准",
                                    "estimated_cost": "以实际票价为准",
                                    "source_refs": ["evidence_1"],
                                    "notes": "避开高峰",
                                }
                            ],
                        }
                    ],
                    "total_budget": "以实际票价为准",
                    "assumptions": ["默认一日游"],
                    "warnings": [],
                },
                "itinerary_validation": {
                    "valid": True,
                    "issues": [],
                    "stats": {
                        "day_count": 1,
                        "slot_count": 1,
                        "slots_with_sources": 1,
                    },
                },
                "agent_trace": {
                    "planner_latency_ms": 10,
                    "execution_latency_ms": 20,
                    "reporter_latency_ms": 30,
                    "plan_mode": "detailed_itinerary",
                    "task_count": 1,
                    "cache_hits": 0,
                    "cache_misses": 1,
                    "tasks": [
                        {
                            "task_id": "task-page",
                            "task_type": "local_search",
                            "status": "completed",
                            "latency_ms": 12,
                            "tool_name": "local_search",
                            "source_count": 2,
                            "cache_hit": False,
                            "error": None,
                        }
                    ],
                },
            }
        ]

        with (
            patch.object(dashboard_router, "summarize_traces", return_value=summary) as mock_summary,
            patch.object(dashboard_router, "list_traces", return_value=traces) as mock_list,
        ):
            response = self.client.get("/dashboard?summary_limit=10&trace_limit=5")

        self.assertEqual(response.status_code, 200)
        self.assertIn("text/html", response.headers["content-type"])
        self.assertIn("GraphRAG Dashboard", response.text)
        self.assertIn("Total Requests", response.text)
        self.assertIn("trace-page", response.text)
        self.assertIn("NEO4J_UNAVAILABLE", response.text)
        self.assertIn("Planner", response.text)
        self.assertIn("Execution", response.text)
        self.assertIn("Reporter", response.text)
        self.assertIn("detailed_itinerary", response.text)
        self.assertIn("Itinerary Preview", response.text)
        self.assertIn("象鼻山", response.text)
        self.assertIn("Valid", response.text)
        self.assertIn("evidence_1", response.text)
        self.assertIn("task-page", response.text)
        self.assertIn("local_search", response.text)
        mock_summary.assert_called_once_with(limit=10)
        mock_list.assert_called_once_with(limit=5)

    def test_traces_returns_recent_trace_list(self):
        traces = [
            {
                "trace_id": "trace-2",
                "requested_route": "hybrid",
                "actual_route": "vector",
                "latency_ms": 45,
                "success": True,
                "degraded": True,
            },
            {
                "trace_id": "trace-1",
                "requested_route": "vector",
                "actual_route": "vector",
                "latency_ms": 30,
                "success": True,
                "degraded": False,
            },
        ]

        with patch.object(dashboard_router, "list_traces", return_value=traces) as mock_list:
            response = self.client.get("/dashboard/traces?limit=2")

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json(), {"count": 2, "traces": traces})
        mock_list.assert_called_once_with(limit=2)

    def test_trace_detail_returns_single_trace(self):
        trace = {
            "trace_id": "trace-1",
            "question": "测试问题",
            "requested_route": "vector",
            "actual_route": "vector",
            "latency_ms": 30,
            "success": True,
            "error_code": None,
            "source_count": 1,
            "degraded": False,
        }

        with patch.object(dashboard_router, "get_trace", return_value=trace) as mock_get:
            response = self.client.get("/dashboard/traces/trace-1")

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json(), trace)
        mock_get.assert_called_once_with("trace-1")

    def test_trace_detail_returns_404_for_missing_trace(self):
        with patch.object(dashboard_router, "get_trace", return_value=None):
            response = self.client.get("/dashboard/traces/missing")

        self.assertEqual(response.status_code, 404)
        self.assertEqual(response.json()["detail"], "Trace not found: missing")


if __name__ == "__main__":
    unittest.main()

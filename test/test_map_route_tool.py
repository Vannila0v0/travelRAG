import json
import os
import unittest
from unittest.mock import patch

from agent_system.core.plan_spec import TaskNode
from agent_system.core.state import PlanExecuteState
from agent_system.executor.retrieval_executor import RetrievalExecutor
from agent_system.executor.tool_registry import MapRouteTool, TOOL_REGISTRY


class FakeAmapResponse:
    def __init__(self, payload):
        self._payload = payload

    def raise_for_status(self):
        return None

    def json(self):
        return self._payload


class MapRouteToolTest(unittest.TestCase):
    def tearDown(self):
        for key in [
            "MAP_ROUTE_PROVIDER",
            "AMAP_API_KEY",
            "AMAP_DEFAULT_CITY",
            "MAP_ROUTE_DEFAULT_MODE",
            "MAP_ROUTE_DEFAULT_ORIGIN",
            "MAP_ROUTE_DEFAULT_VISIT_DURATION_MIN",
            "MAP_ROUTE_MAX_EXACT_PLACES",
            "MAP_ROUTE_ROAD_FACTOR",
        ]:
            os.environ.pop(key, None)

    def test_registry_returns_map_route_tool(self):
        tool = TOOL_REGISTRY["map_route"]()

        self.assertIsInstance(tool, MapRouteTool)

    def test_map_route_optimizes_multi_place_order(self):
        result = MapRouteTool().structured_search(
            {
                "origin": "桂林站",
                "destination": "靖江王府",
                "mode": "taxi",
                "start_time": "09:00",
                "end_time": "15:00",
                "places": [
                    {"name": "靖江王府", "visit_duration_min": 90},
                    {"name": "象鼻山", "visit_duration_min": 60},
                    {"name": "东西巷", "visit_duration_min": 60},
                ],
                "constraints": {"pace": "relaxed"},
            }
        )

        self.assertTrue(result["success"])
        payload = json.loads(result["answer"])
        self.assertEqual(payload["provider"], "local")
        self.assertEqual(payload["mode"], "taxi")
        self.assertEqual(payload["route_order"][0], "桂林站")
        self.assertEqual(payload["route_order"][1], "象鼻山")
        self.assertEqual(payload["route_order"][-1], "靖江王府")
        self.assertTrue(payload["feasible"])
        self.assertGreater(payload["total_travel_time_min"], 0)
        self.assertEqual(result["data"]["route"], "map_route")
        self.assertEqual(result["data"]["sources"][0]["source_type"], "route_plan")

    def test_map_route_extracts_known_places_from_query_text(self):
        result = MapRouteTool().structured_search(
            {
                "query": "从桂林站出发，象鼻山、靖江王府、东西巷怎么走比较不绕路？",
            }
        )

        self.assertTrue(result["success"])
        payload = json.loads(result["answer"])
        self.assertEqual(payload["origin"], "桂林站")
        self.assertEqual(payload["route_order"][0], "桂林站")
        self.assertEqual(payload["route_order"].count("桂林站"), 1)
        self.assertIn("象鼻山", payload["route_order"])
        self.assertIn("靖江王府", payload["route_order"])
        self.assertIn("东西巷", payload["route_order"])
        self.assertEqual(payload["total_visit_time_min"], 270)

    def test_map_route_extracts_places_from_previous_records(self):
        result = MapRouteTool().structured_search(
            {
                "query": "帮我规划桂林市区一日游路线",
                "origin": "桂林站",
                "previous_records": [
                    {
                        "route": "global",
                        "output": "桂林市区一日游可以考虑象鼻山、靖江王府和东西巷，路线要减少往返。",
                    }
                ],
            }
        )

        self.assertTrue(result["success"])
        payload = json.loads(result["answer"])
        self.assertIn("象鼻山", payload["route_order"])
        self.assertIn("靖江王府", payload["route_order"])
        self.assertIn("东西巷", payload["route_order"])

    def test_map_route_returns_missing_coordinates_error(self):
        result = MapRouteTool().structured_search(
            {
                "origin": "桂林站",
                "places": ["不存在的景点"],
            }
        )

        self.assertFalse(result["success"])
        self.assertEqual(result["data"]["metadata"]["reason"], "missing_coordinates")
        self.assertIn("不存在的景点", result["data"]["metadata"]["missing_places"])

    def test_retrieval_executor_can_run_map_route_task(self):
        task = TaskNode(
            task_id="task-route",
            task_type="map_route",
            description="优化多个景点顺序",
            parameters={
                "origin": "桂林站",
                "places": ["象鼻山", "东西巷", "靖江王府"],
            },
        )

        record = RetrievalExecutor().execute_task(task, PlanExecuteState(session_id="s", input_query="q"))

        self.assertEqual(record.route, "map_route")
        self.assertIsNone(record.metadata.error)
        self.assertEqual(record.sources[0]["source_type"], "route_plan")
        self.assertIn("象鼻山", record.output)

    @patch("agent_system.executor.tool_registry.httpx.get")
    def test_amap_provider_geocodes_places_and_uses_driving_route(self, mock_get):
        os.environ["MAP_ROUTE_PROVIDER"] = "amap"
        os.environ["AMAP_API_KEY"] = "test-key"
        os.environ["AMAP_DEFAULT_CITY"] = "桂林"

        locations = {
            "起点酒店": "110.290000,25.270000",
            "测试景点A": "110.300000,25.280000",
            "测试景点B": "110.310000,25.290000",
        }

        def fake_get(url, params=None, timeout=None):
            params = params or {}
            if "geocode/geo" in url:
                return FakeAmapResponse(
                    {
                        "status": "1",
                        "info": "OK",
                        "geocodes": [
                            {
                                "formatted_address": params["address"],
                                "location": locations[params["address"]],
                                "level": "兴趣点",
                            }
                        ],
                    }
                )
            if "direction/driving" in url:
                return FakeAmapResponse(
                    {
                        "status": "1",
                        "info": "OK",
                        "route": {
                            "paths": [
                                {
                                    "distance": "1200",
                                    "duration": "600",
                                }
                            ]
                        },
                    }
                )
            raise AssertionError(f"unexpected amap URL: {url}")

        mock_get.side_effect = fake_get

        result = MapRouteTool().structured_search(
            {
                "origin": "起点酒店",
                "places": ["测试景点A", "测试景点B"],
                "mode": "taxi",
                "start_time": "09:00",
                "end_time": "13:00",
            }
        )

        self.assertTrue(result["success"])
        payload = json.loads(result["answer"])
        self.assertEqual(payload["provider"], "amap")
        self.assertEqual(payload["total_travel_time_min"], 20)
        self.assertTrue(all(leg["provider"] == "amap" for leg in payload["legs"]))
        self.assertTrue(any("geocode/geo" in call.args[0] for call in mock_get.call_args_list))
        self.assertTrue(any("direction/driving" in call.args[0] for call in mock_get.call_args_list))


if __name__ == "__main__":
    unittest.main()

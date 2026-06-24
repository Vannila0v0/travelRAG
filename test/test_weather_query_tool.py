import json
import os
import unittest
from datetime import date, timedelta
from unittest.mock import patch

from agent_system.core.plan_spec import TaskNode
from agent_system.core.state import PlanExecuteState
from agent_system.executor.retrieval_executor import RetrievalExecutor
from agent_system.executor.tool_registry import TOOL_REGISTRY, WeatherQueryTool


class FakeJsonResponse:
    def __init__(self, payload, status_code=200):
        self._payload = payload
        self.status_code = status_code
        self.headers = {"content-type": "application/json"}
        self.text = json.dumps(payload)

    def raise_for_status(self):
        if self.status_code >= 400:
            raise RuntimeError(f"status={self.status_code}")

    def json(self):
        return self._payload


class WeatherQueryToolTest(unittest.TestCase):
    def tearDown(self):
        for key in [
            "WEATHER_QUERY_PROVIDER",
            "WEATHER_PROVIDER",
            "WEATHER_QUERY_MOCK_RESULT",
            "WEATHER_QUERY_MAX_FORECAST_DAYS",
            "WEATHER_QUERY_DEFAULT_LOCATION",
            "OPEN_METEO_GEOCODING_ENDPOINT",
            "OPEN_METEO_FORECAST_ENDPOINT",
        ]:
            os.environ.pop(key, None)

    def test_registry_returns_weather_query_tool(self):
        tool = TOOL_REGISTRY["weather_query"]()

        self.assertIsInstance(tool, WeatherQueryTool)

    def test_mock_weather_returns_structured_forecast(self):
        os.environ["WEATHER_QUERY_PROVIDER"] = "mock"
        start = date.today() + timedelta(days=1)
        end = start + timedelta(days=1)

        result = WeatherQueryTool().structured_search(
            {
                "location": "Guilin",
                "start_date": start.isoformat(),
                "end_date": end.isoformat(),
                "query": "Guilin two day trip",
            }
        )

        self.assertTrue(result["success"])
        payload = json.loads(result["answer"])
        self.assertEqual(payload["provider"], "mock")
        self.assertEqual(payload["location"], "Guilin")
        self.assertEqual(len(payload["daily"]), 2)
        self.assertEqual(result["data"]["route"], "weather_query")
        self.assertEqual(result["data"]["metadata"]["result_count"], 2)
        self.assertEqual(result["data"]["sources"][0]["source_type"], "weather")

    def test_open_meteo_provider_queries_geocoding_and_forecast(self):
        start = date.today() + timedelta(days=1)
        geocoding_response = FakeJsonResponse(
            {
                "results": [
                    {
                        "name": "Guilin",
                        "country": "China",
                        "admin1": "Guangxi",
                        "latitude": 25.28,
                        "longitude": 110.29,
                        "timezone": "Asia/Shanghai",
                    }
                ]
            }
        )
        forecast_response = FakeJsonResponse(
            {
                "daily": {
                    "time": [start.isoformat()],
                    "weather_code": [61],
                    "temperature_2m_max": [29.5],
                    "temperature_2m_min": [23.0],
                    "precipitation_probability_max": [80],
                    "precipitation_sum": [8.2],
                    "wind_speed_10m_max": [12.0],
                }
            }
        )

        with patch(
            "agent_system.executor.tool_registry.httpx.get",
            side_effect=[geocoding_response, forecast_response],
        ) as get:
            result = WeatherQueryTool().structured_search(
                {
                    "location": "Guilin",
                    "start_date": start.isoformat(),
                    "end_date": start.isoformat(),
                }
            )

        self.assertTrue(result["success"])
        payload = json.loads(result["answer"])
        self.assertEqual(payload["provider"], "open_meteo")
        self.assertEqual(payload["daily"][0]["condition"], "rain")
        self.assertIn("rain_sensitive", payload["daily"][0]["travel_impact"])
        self.assertEqual(get.call_count, 2)
        self.assertEqual(get.call_args_list[0].kwargs["params"]["name"], "Guilin")
        self.assertEqual(get.call_args_list[1].kwargs["params"]["start_date"], start.isoformat())
        self.assertEqual(get.call_args_list[1].kwargs["params"]["end_date"], start.isoformat())

    def test_forecast_range_exceeded_returns_planning_limitation(self):
        start = date.today() + timedelta(days=60)

        result = WeatherQueryTool().structured_search(
            {
                "location": "Guilin",
                "start_date": start.isoformat(),
                "end_date": start.isoformat(),
            }
        )

        self.assertTrue(result["success"])
        payload = json.loads(result["answer"])
        self.assertFalse(payload["forecast_available"])
        self.assertEqual(result["data"]["metadata"]["reason"], "forecast_range_exceeded")

    def test_retrieval_executor_can_run_weather_query_task(self):
        os.environ["WEATHER_QUERY_PROVIDER"] = "mock"
        start = date.today() + timedelta(days=1)
        task = TaskNode(
            task_id="task-weather",
            task_type="weather_query",
            description="Weather-aware trip plan",
            parameters={
                "location": "Guilin",
                "start_date": start.isoformat(),
                "end_date": start.isoformat(),
            },
        )

        record = RetrievalExecutor().execute_task(task, PlanExecuteState(session_id="s", input_query="q"))

        self.assertEqual(record.route, "weather_query")
        self.assertEqual(record.tool_metadata["provider"], "mock")
        self.assertEqual(record.sources[0]["source_type"], "weather")
        self.assertIsNone(record.metadata.error)


if __name__ == "__main__":
    unittest.main()

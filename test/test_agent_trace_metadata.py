import unittest

from agent_system.core.execution_record import ExecutionMetadata, ExecutionRecord
from agent_system.core.plan_spec import PlanSpec, TaskGraph, TaskNode
from agent_system.core.state import PlanExecuteState
from query_engine.router import _build_agent_metadata


class AgentTraceMetadataTest(unittest.TestCase):
    def test_build_agent_metadata_includes_stage_and_task_trace(self):
        task = TaskNode(
            task_id="task-1",
            task_type="local_search",
            description="查找票价",
            status="completed",
        )
        state = PlanExecuteState(
            session_id="session-1",
            input_query="测试问题",
            plan_mode="place_recommendations",
        )
        state.plan = PlanSpec(
            original_query=state.input_query,
            task_graph=TaskGraph(nodes=[task]),
        )
        state.agent_trace = {
            "planner_latency_ms": 10,
            "execution_latency_ms": 20,
            "reporter_latency_ms": 30,
            "task_count": 1,
        }
        state.execution_records.append(
            ExecutionRecord(
                task_id="task-1",
                worker_type="retrieval_executor",
                inputs={"task_type": "local_search"},
                output="answer",
                route="hybrid",
                sources=[{"chunk_id": "chunk-1"}, {"chunk_id": "chunk-2"}],
                tool_metadata={"cache_hit": False},
                metadata=ExecutionMetadata(
                    worker_type="retrieval_executor",
                    latency_seconds=0.123,
                    error=None,
                ),
            )
        )

        metadata = _build_agent_metadata(state)
        agent_trace = metadata["agent_trace"]

        self.assertEqual(metadata["plan_mode"], "place_recommendations")
        self.assertEqual(agent_trace["planner_latency_ms"], 10)
        self.assertEqual(agent_trace["execution_latency_ms"], 20)
        self.assertEqual(agent_trace["reporter_latency_ms"], 30)
        self.assertEqual(agent_trace["task_count"], 1)
        self.assertEqual(agent_trace["cache_hits"], 0)
        self.assertEqual(agent_trace["cache_misses"], 1)
        self.assertEqual(agent_trace["tasks"][0]["task_id"], "task-1")
        self.assertEqual(agent_trace["tasks"][0]["task_type"], "local_search")
        self.assertEqual(agent_trace["tasks"][0]["status"], "completed")
        self.assertEqual(agent_trace["tasks"][0]["latency_ms"], 123)
        self.assertEqual(agent_trace["tasks"][0]["tool_name"], "local_search")
        self.assertEqual(agent_trace["tasks"][0]["source_count"], 2)
        self.assertEqual(agent_trace["tasks"][0]["route"], "hybrid")
        self.assertFalse(agent_trace["tasks"][0]["cache_hit"])

        self.assertEqual(metadata["tool_cache"], {"hits": 0, "misses": 1, "total": 1})
        self.assertEqual(metadata["tasks"][0]["task_id"], "task-1")


if __name__ == "__main__":
    unittest.main()

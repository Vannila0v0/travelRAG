import unittest

from agent_system.core.plan_spec import TaskGraph, TaskNode
from agent_system.planner.plan_compactor import PlanCompactor
from agent_system.planner.task_normalizer import TaskNormalizer


def node(task_id, task_type, description, *, depends_on=None, entities=None, priority=2, parameters=None):
    return TaskNode(
        task_id=task_id,
        task_type=task_type,
        description=description,
        priority=priority,
        depends_on=depends_on or [],
        entities=entities or [],
        parameters=parameters or {},
    )


class PlanCompactorTest(unittest.TestCase):
    def test_compactor_merges_mergeable_tools_and_rebuilds_dependencies(self):
        graph = TaskGraph(
            nodes=[
                node("g1", "global_search", "overall route", priority=1),
                node("g2", "global_search", "city level recommendation", priority=2),
                node("l1", "local_search", "ticket details", depends_on=["g1"], entities=["Elephant Hill"]),
                node("l2", "local_search", "traffic details", depends_on=["g1"], entities=["Li River"]),
                node("l3", "local_search", "opening hours", depends_on=["g2"], entities=["West Street"]),
                node(
                    "w1",
                    "weather_query",
                    "weather",
                    parameters={"location": "Guilin", "start_date": "2026-06-23"},
                ),
                node(
                    "r1",
                    "reflection",
                    "synthesis",
                    depends_on=["g1", "g2", "l1", "l2", "l3", "w1"],
                ),
            ],
            execution_mode="sequential",
        )

        compactor = PlanCompactor(max_tasks=4)
        compacted = compactor.compact(graph)

        self.assertEqual([item.task_type for item in compacted.nodes], [
            "global_search",
            "local_search",
            "weather_query",
            "reflection",
        ])
        self.assertEqual(len(compacted.nodes), 4)
        self.assertEqual(compacted.nodes[1].entities, ["Elephant Hill", "Li River", "West Street"])
        self.assertEqual(compacted.nodes[3].depends_on, ["task_001", "task_002", "task_003"])
        self.assertEqual(compactor.last_trace["original_task_count"], 7)
        self.assertEqual(compactor.last_trace["final_task_count"], 4)
        self.assertTrue(compactor.last_trace["merged_tasks"])

    def test_task_normalizer_compacts_generic_llm_plan_to_max_tasks(self):
        graph = TaskGraph(
            nodes=[
                node("raw_1", "global_search", "overall plan"),
                node("raw_2", "global_search", "route frame"),
                node("raw_3", "local_search", "spot A", entities=["A"]),
                node("raw_4", "local_search", "spot B", entities=["B"]),
                node("raw_5", "local_search", "spot C", entities=["C"]),
                node("raw_6", "local_search", "spot D", entities=["D"]),
                node("raw_7", "reflection", "synthesis", depends_on=["raw_1", "raw_3", "raw_4"]),
            ],
            execution_mode="sequential",
        )

        normalized = TaskNormalizer(max_tasks=3).normalize("ordinary trip planning request", graph)

        self.assertLessEqual(len(normalized.nodes), 3)
        self.assertEqual(normalized.nodes[-1].task_type, "reflection")
        self.assertEqual(normalized.nodes[-1].depends_on, ["task_001", "task_002"])


if __name__ == "__main__":
    unittest.main()

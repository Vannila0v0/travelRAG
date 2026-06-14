from __future__ import annotations

from ..core.plan_spec import TaskGraph, TaskNode
from .task_template_matcher import TaskTemplateMatcher


class TaskNormalizer:
    """Normalize LLM-generated task graphs into stable executable DAGs."""

    def __init__(self, template_matcher: TaskTemplateMatcher | None = None):
        self._template_matcher = template_matcher or TaskTemplateMatcher()

    def normalize(self, query: str, task_graph: TaskGraph) -> TaskGraph:
        template_graph = self._template_matcher.match(query)
        if template_graph:
            return template_graph
        return self._normalize_generic(task_graph)

    def _normalize_generic(self, task_graph: TaskGraph) -> TaskGraph:
        normalized_nodes: list[TaskNode] = []
        id_map: dict[str, str] = {}

        for index, node in enumerate(task_graph.nodes, start=1):
            new_id = f"task_{index:03d}"
            id_map[node.task_id] = new_id

        for index, node in enumerate(task_graph.nodes, start=1):
            new_id = f"task_{index:03d}"
            depends_on = [
                id_map[dep]
                for dep in node.depends_on
                if dep in id_map and id_map[dep] != new_id
            ]
            normalized_nodes.append(
                TaskNode(
                    task_id=new_id,
                    task_type=node.task_type,
                    description=node.description.strip(),
                    priority=node.priority,
                    depends_on=depends_on,
                    entities=self._dedupe_texts(node.entities),
                    parameters=dict(node.parameters),
                    status="pending",
                )
            )

        return TaskGraph(
            nodes=normalized_nodes,
            execution_mode=task_graph.execution_mode,
        )

    @staticmethod
    def _dedupe_texts(values: list[str]) -> list[str]:
        seen = set()
        result = []
        for value in values or []:
            text = str(value).strip()
            if not text or text in seen:
                continue
            seen.add(text)
            result.append(text)
        return result

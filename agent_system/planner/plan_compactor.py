from __future__ import annotations

from collections import defaultdict
from typing import Any

from ..core.plan_spec import TaskGraph, TaskNode
from ..executor.tool_registry import PlanPolicy, TOOL_SPECS


UNKNOWN_TOOL_POLICY = PlanPolicy(
    importance="required",
    merge_strategy="none",
    max_instances=None,
    drop_priority=20,
)


class PlanCompactor:
    """Compact oversized task graphs using plan policies registered with tools."""

    def __init__(self, max_tasks: int = 5, tool_specs: dict[str, Any] | None = None):
        self.max_tasks = max(1, int(max_tasks))
        self.tool_specs = tool_specs or TOOL_SPECS
        self.last_trace: dict[str, Any] = {}

    def compact(self, task_graph: TaskGraph) -> TaskGraph:
        self.last_trace = {
            "max_tasks": self.max_tasks,
            "original_task_count": len(task_graph.nodes),
            "final_task_count": len(task_graph.nodes),
            "compacted": False,
            "merged_tasks": [],
            "removed_tasks": [],
        }
        if len(task_graph.nodes) <= self.max_tasks:
            return task_graph

        replacements: dict[str, str] = {}
        removed_ids: set[str] = set()
        nodes = list(task_graph.nodes)

        nodes, merge_replacements, merged_trace = self._merge_by_policy(nodes)
        replacements.update(merge_replacements)
        self.last_trace["merged_tasks"].extend(merged_trace)

        nodes, removed = self._enforce_max_instances(nodes)
        removed_ids.update(item["task_id"] for item in removed)
        self.last_trace["removed_tasks"].extend(removed)

        if len(nodes) > self.max_tasks:
            nodes, removed = self._drop_to_limit(nodes)
            removed_ids.update(item["task_id"] for item in removed)
            self.last_trace["removed_tasks"].extend(removed)

        compacted = self._rebuild_graph(
            task_graph=task_graph,
            nodes=nodes,
            replacements=replacements,
            removed_ids=removed_ids,
        )
        self.last_trace["final_task_count"] = len(compacted.nodes)
        self.last_trace["compacted"] = len(compacted.nodes) != self.last_trace["original_task_count"]
        return compacted

    def _merge_by_policy(
        self,
        nodes: list[TaskNode],
    ) -> tuple[list[TaskNode], dict[str, str], list[dict[str, Any]]]:
        groups: dict[str, list[TaskNode]] = defaultdict(list)
        for node in nodes:
            policy = self._policy(node.task_type)
            if policy.merge_strategy in {"same_tool_merge_entities", "same_tool_merge_query"}:
                groups[node.task_type].append(node)

        replacements: dict[str, str] = {}
        merged_nodes: dict[str, TaskNode] = {}
        trace: list[dict[str, Any]] = []
        for task_type, group in groups.items():
            if len(group) <= 1:
                continue
            merged = self._merge_nodes(group)
            merged_nodes[group[0].task_id] = merged
            for node in group:
                replacements[node.task_id] = merged.task_id
            trace.append(
                {
                    "from": [node.task_id for node in group],
                    "to": merged.task_id,
                    "task_type": task_type,
                    "strategy": self._policy(task_type).merge_strategy,
                }
            )

        if not merged_nodes:
            return nodes, replacements, trace

        output: list[TaskNode] = []
        skipped = {old_id for old_id, new_id in replacements.items() if old_id != new_id}
        for node in nodes:
            if node.task_id in merged_nodes:
                output.append(merged_nodes[node.task_id])
                continue
            if node.task_id in skipped:
                continue
            output.append(node)
        return output, replacements, trace

    def _merge_nodes(self, nodes: list[TaskNode]) -> TaskNode:
        first = nodes[0]
        descriptions = []
        entities: list[str] = []
        depends_on: list[str] = []
        parameters: dict[str, Any] = dict(first.parameters)
        source_ids = [node.task_id for node in nodes]
        source_id_set = set(source_ids)

        for node in nodes:
            description = str(node.description or "").strip()
            if description and description not in descriptions:
                descriptions.append(description)
            for entity in node.entities:
                if entity not in entities:
                    entities.append(entity)
            for dep in node.depends_on:
                if dep not in source_id_set and dep not in depends_on:
                    depends_on.append(dep)
            for key, value in node.parameters.items():
                if key not in parameters:
                    parameters[key] = value

        parameters["_compacted_from"] = source_ids
        return TaskNode(
            task_id=first.task_id,
            task_type=first.task_type,
            description="; ".join(descriptions)[:2000] or first.description,
            priority=min(node.priority for node in nodes),
            depends_on=depends_on,
            entities=entities,
            parameters=parameters,
            status="pending",
        )

    def _enforce_max_instances(self, nodes: list[TaskNode]) -> tuple[list[TaskNode], list[dict[str, Any]]]:
        by_type: dict[str, list[tuple[int, TaskNode]]] = defaultdict(list)
        for index, node in enumerate(nodes):
            by_type[node.task_type].append((index, node))

        remove_ids: set[str] = set()
        removed: list[dict[str, Any]] = []
        for task_type, indexed_nodes in by_type.items():
            policy = self._policy(task_type)
            if policy.max_instances is None or len(indexed_nodes) <= policy.max_instances:
                continue
            keep = sorted(indexed_nodes, key=lambda item: (item[1].priority, item[0]))[: policy.max_instances]
            keep_ids = {node.task_id for _, node in keep}
            for _, node in indexed_nodes:
                if node.task_id in keep_ids:
                    continue
                remove_ids.add(node.task_id)
                removed.append(
                    {
                        "task_id": node.task_id,
                        "task_type": node.task_type,
                        "reason": f"max_instances={policy.max_instances}",
                    }
                )

        return [node for node in nodes if node.task_id not in remove_ids], removed

    def _drop_to_limit(self, nodes: list[TaskNode]) -> tuple[list[TaskNode], list[dict[str, Any]]]:
        output = list(nodes)
        removed: list[dict[str, Any]] = []

        while len(output) > self.max_tasks:
            candidates = [
                (self._drop_score(node, index), index, node)
                for index, node in enumerate(output)
                if self._policy(node.task_type).importance != "critical"
            ]
            if not candidates:
                candidates = [
                    (self._drop_score(node, index), index, node)
                    for index, node in enumerate(output)
                ]
            _, index, node = max(candidates, key=lambda item: item[0])
            output.pop(index)
            removed.append(
                {
                    "task_id": node.task_id,
                    "task_type": node.task_type,
                    "reason": "task_count_limit",
                }
            )

        return output, removed

    def _rebuild_graph(
        self,
        *,
        task_graph: TaskGraph,
        nodes: list[TaskNode],
        replacements: dict[str, str],
        removed_ids: set[str],
    ) -> TaskGraph:
        final_ids = {node.task_id for node in nodes}
        old_to_new = {node.task_id: f"task_{index:03d}" for index, node in enumerate(nodes, start=1)}
        rebuilt_nodes: list[TaskNode] = []

        for node in nodes:
            resolved_deps: list[str] = []
            for dep in node.depends_on:
                resolved = replacements.get(dep, dep)
                if resolved in removed_ids or resolved not in final_ids or resolved == node.task_id:
                    continue
                new_dep = old_to_new[resolved]
                if new_dep not in resolved_deps:
                    resolved_deps.append(new_dep)

            rebuilt_nodes.append(
                TaskNode(
                    task_id=old_to_new[node.task_id],
                    task_type=node.task_type,
                    description=node.description,
                    priority=node.priority,
                    depends_on=resolved_deps,
                    entities=list(node.entities),
                    parameters=dict(node.parameters),
                    status="pending",
                )
            )

        return TaskGraph(nodes=rebuilt_nodes, execution_mode=task_graph.execution_mode)

    def _policy(self, task_type: str) -> PlanPolicy:
        spec = self.tool_specs.get(task_type)
        if spec is None:
            return UNKNOWN_TOOL_POLICY
        return getattr(spec, "plan_policy", UNKNOWN_TOOL_POLICY)

    def _drop_score(self, node: TaskNode, index: int) -> tuple[int, int, int, int]:
        policy = self._policy(node.task_type)
        importance_score = {
            "critical": 0,
            "required": 1,
            "mergeable": 2,
            "normal": 3,
            "optional": 4,
        }.get(policy.importance, 3)
        return importance_score, policy.drop_priority, node.priority, index

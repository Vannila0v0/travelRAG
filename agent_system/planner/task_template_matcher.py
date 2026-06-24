from __future__ import annotations

import json
import logging
import os
import re
from pathlib import Path
from typing import Any

from ..core.plan_spec import TaskGraph, TaskNode

_LOGGER = logging.getLogger(__name__)


class TaskTemplateMatcher:
    """Load configured task DAG templates and match them against user queries."""

    def __init__(self, config_path: str | Path | None = None):
        self._config_path = self._resolve_config_path(config_path)
        self._templates = self._load_templates(self._config_path)

    def match(self, query: str) -> TaskGraph | None:
        for template in self._templates:
            if self._matches_rule(query, template.get("match", {})):
                try:
                    return self._build_task_graph(template, query=query)
                except Exception as exc:
                    _LOGGER.warning(
                        "Matched task template %s but failed to build graph: %s",
                        template.get("name", "<unnamed>"),
                        exc,
                    )
                    continue
        return None

    @staticmethod
    def _resolve_config_path(config_path: str | Path | None) -> Path:
        if config_path:
            return Path(config_path)

        env_path = os.getenv("TASK_TEMPLATE_CONFIG")
        if env_path:
            return Path(env_path)

        project_root = Path(__file__).resolve().parents[2]
        return project_root / "configs" / "task_templates.json"

    @staticmethod
    def _load_templates(config_path: Path) -> list[dict[str, Any]]:
        if not config_path.exists():
            _LOGGER.warning("Task template config not found: %s", config_path)
            return []

        try:
            data = json.loads(config_path.read_text(encoding="utf-8"))
        except Exception as exc:
            _LOGGER.warning("Failed to load task template config %s: %s", config_path, exc)
            return []

        templates = data.get("templates", [])
        if not isinstance(templates, list):
            _LOGGER.warning("Invalid task template config: templates must be a list")
            return []
        return [template for template in templates if isinstance(template, dict)]

    def _build_task_graph(self, template: dict[str, Any], query: str = "") -> TaskGraph:
        nodes = []
        for raw_node in template.get("nodes", []):
            if not isinstance(raw_node, dict):
                continue

            node_data = self._render_value(dict(raw_node), query)
            node_data.setdefault("depends_on", [])
            node_data.setdefault("entities", [])
            node_data.setdefault("parameters", {})
            node_data.setdefault("status", "pending")
            nodes.append(TaskNode(**node_data))

        if not nodes:
            raise ValueError("task template has no valid nodes")

        return TaskGraph(
            nodes=nodes,
            execution_mode=template.get("execution_mode", "sequential"),
        )

    def _render_value(self, value: Any, query: str) -> Any:
        if isinstance(value, str):
            return value.replace("{query}", query)
        if isinstance(value, list):
            return [self._render_value(item, query) for item in value]
        if isinstance(value, dict):
            return {key: self._render_value(item, query) for key, item in value.items()}
        return value

    def _matches_rule(self, query: str, rule: dict[str, Any]) -> bool:
        if not rule:
            return False

        all_groups = rule.get("all", [])
        if all_groups and not all(self._matches_group(query, group) for group in all_groups):
            return False

        any_keywords = rule.get("any", [])
        if any_keywords and not self._has_any(query, any_keywords):
            return False

        not_keywords = rule.get("not", [])
        if not_keywords and self._has_any(query, not_keywords):
            return False

        regex = rule.get("regex")
        if regex and not re.search(str(regex), query):
            return False

        return True

    def _matches_group(self, query: str, group: dict[str, Any]) -> bool:
        any_keywords = group.get("any", [])
        if any_keywords and not self._has_any(query, any_keywords):
            return False

        all_keywords = group.get("all", [])
        if all_keywords and not all(str(keyword) in query for keyword in all_keywords):
            return False

        not_keywords = group.get("not", [])
        if not_keywords and self._has_any(query, not_keywords):
            return False

        return True

    @staticmethod
    def _has_any(text: str, keywords: list[Any]) -> bool:
        return any(str(keyword) in text for keyword in keywords)

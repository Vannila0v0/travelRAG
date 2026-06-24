from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from .markdown_skill import MarkdownSkillSpec


@dataclass(frozen=True)
class ClarificationResult:
    answer: str
    metadata: dict


class TravelPlanClarifierSkill:
    """Execute the markdown-defined travel plan clarification skill."""

    def __init__(self, spec_path: str | Path | None = None):
        self.spec = MarkdownSkillSpec.load(
            spec_path or Path(__file__).with_name("travel_plan_clarifier.md")
        )
        self.name = self.spec.name
        self.config = self.spec.config

    def match(
        self,
        question: str,
        route: str | None = None,
        plan_mode: str = "auto",
    ) -> ClarificationResult | None:
        if plan_mode != "auto":
            return None

        route_scope = set(self.config.get("route_scope", []))
        if route is not None and route_scope and route not in route_scope:
            return None

        normalized = "".join(question.split())
        if not normalized:
            return None

        if not self._looks_like_whole_trip_request(normalized):
            return None

        if self._already_chooses_mode(normalized):
            return None

        return ClarificationResult(
            answer="\n".join(self.config.get("clarification_answer_lines", [])),
            metadata={
                "skill": self.name,
                "skill_spec": self._relative_spec_path(),
                "clarification_required": True,
                "clarification_type": self.config.get("clarification_type", "travel_plan_mode"),
                "options": self.config.get("options", []),
            },
        )

    def _looks_like_whole_trip_request(self, text: str) -> bool:
        has_trip_term = self._contains_any(text, "trigger_terms")
        has_planning_verb = self._contains_any(text, "planning_verbs")
        broad_destination = self._contains_any(text, "broad_destinations")
        return has_trip_term and (has_planning_verb or broad_destination)

    def _already_chooses_mode(self, text: str) -> bool:
        return (
            self._contains_any(text, "detail_mode_terms")
            or self._contains_any(text, "recommendation_mode_terms")
        )

    def _contains_any(self, text: str, config_key: str) -> bool:
        return any(term in text for term in self.config.get(config_key, []))

    def _relative_spec_path(self) -> str:
        try:
            return self.spec.path.relative_to(Path.cwd()).as_posix()
        except ValueError:
            return self.spec.path.as_posix()

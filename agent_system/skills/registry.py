from __future__ import annotations

from .travel_plan_clarifier import ClarificationResult, TravelPlanClarifierSkill


PRE_QUERY_SKILLS = (
    TravelPlanClarifierSkill(),
)


def match_pre_query_skill(
    question: str,
    route: str,
    plan_mode: str = "auto",
) -> ClarificationResult | None:
    for skill in PRE_QUERY_SKILLS:
        result = skill.match(question, route=route, plan_mode=plan_mode)
        if result is not None:
            return result
    return None

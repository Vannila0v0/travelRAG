import json
import logging
import re
import time
from typing import Any

from ..config.prompts.reporter_prompts import ITINERARY_JSON_PROMPT
from ..core.state import PlanExecuteState
from ..integration.llm_factory import get_llm_model


_LOGGER = logging.getLogger(__name__)


class ItineraryBuilder:
    """Build structured itinerary JSON from agent evidence and final report."""

    def __init__(self, llm=None):
        self._llm = llm or get_llm_model()
        self.last_metrics: dict[str, int | str] = {}

    def build(self, state: PlanExecuteState) -> dict[str, Any]:
        started = time.perf_counter()
        evidence_context = self._format_evidence(state)
        prompt = ITINERARY_JSON_PROMPT.format(
            query=state.input_query,
            answer=state.final_report or "",
            evidence_context=evidence_context[:8000],
        )

        _LOGGER.info("[ItineraryBuilder] generating structured itinerary JSON")
        response = self._llm.invoke(prompt)
        content = str(response.content if hasattr(response, "content") else response)
        plan = self._parse_plan(content)
        self.last_metrics = {
            "itinerary_latency_ms": int((time.perf_counter() - started) * 1000),
            "itinerary_evidence_chars": len(evidence_context),
            "itinerary_day_count": len(plan.get("days", [])),
        }
        return plan

    def _parse_plan(self, content: str) -> dict[str, Any]:
        try:
            parsed = json.loads(self._extract_json(content))
        except Exception as exc:
            _LOGGER.error("failed to parse itinerary JSON: %s", exc)
            return {
                "days": [],
                "total_budget": None,
                "assumptions": [],
                "warnings": ["结构化行程解析失败，已保留自然语言答案。"],
            }

        if not isinstance(parsed, dict):
            parsed = {}
        days = parsed.get("days")
        if not isinstance(days, list):
            days = []

        normalized_days = []
        for day_index, day in enumerate(days, start=1):
            if not isinstance(day, dict):
                continue
            slots = day.get("slots")
            if not isinstance(slots, list):
                slots = []
            normalized_slots = []
            for slot in slots:
                if not isinstance(slot, dict):
                    continue
                normalized_slots.append(
                    {
                        "start_time": self._optional_text(slot.get("start_time")),
                        "end_time": self._optional_text(slot.get("end_time")),
                        "title": str(slot.get("title") or "未命名活动"),
                        "location": self._optional_text(slot.get("location")),
                        "activity": self._optional_text(slot.get("activity")),
                        "transport_to_next": self._optional_text(slot.get("transport_to_next")),
                        "estimated_cost": self._optional_text(slot.get("estimated_cost")),
                        "ticket_info": self._optional_text(slot.get("ticket_info")),
                        "source_refs": self._text_list(slot.get("source_refs")),
                        "notes": self._optional_text(slot.get("notes")),
                    }
                )
            normalized_days.append(
                {
                    "date_label": str(day.get("date_label") or f"第 {day_index} 天"),
                    "slots": normalized_slots,
                }
            )

        return {
            "days": normalized_days,
            "total_budget": self._optional_text(parsed.get("total_budget")),
            "assumptions": self._text_list(parsed.get("assumptions")),
            "warnings": self._text_list(parsed.get("warnings")),
        }

    @staticmethod
    def _extract_json(content: str) -> str:
        fenced = re.search(r"```(?:json)?\s*(.*?)```", content, re.DOTALL)
        if fenced:
            return fenced.group(1).strip()
        match = re.search(r"\{.*\}", content, re.DOTALL)
        return match.group(0) if match else content

    @staticmethod
    def _optional_text(value: Any) -> str | None:
        if value is None:
            return None
        text = str(value).strip()
        return text or None

    @staticmethod
    def _text_list(value: Any) -> list[str]:
        if not isinstance(value, list):
            return []
        return [str(item).strip() for item in value if str(item).strip()]

    @staticmethod
    def _format_evidence(state: PlanExecuteState) -> str:
        text = []
        for index, record in enumerate(state.execution_records, start=1):
            route = getattr(record, "route", None) or "unknown"
            output = str(record.output)
            text.append(f"[evidence_{index}] route={route}\n{output}")
        return "\n\n".join(text)


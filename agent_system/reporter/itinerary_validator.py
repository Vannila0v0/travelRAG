from __future__ import annotations

from typing import Any


class ItineraryValidator:
    """Rule-based quality checks for structured itinerary output."""

    def validate(self, plan: dict[str, Any] | None) -> dict[str, Any]:
        issues: list[dict[str, str]] = []
        if not isinstance(plan, dict):
            return {
                "valid": False,
                "issues": [{"code": "missing_plan", "message": "structured_output is missing"}],
                "stats": {"day_count": 0, "slot_count": 0, "slots_with_time": 0, "slots_with_sources": 0},
            }

        days = plan.get("days")
        if not isinstance(days, list) or not days:
            issues.append({"code": "missing_days", "message": "itinerary days must not be empty"})
            days = []

        slot_count = 0
        slots_with_time = 0
        slots_with_sources = 0
        slots_with_detail = 0

        for day_index, day in enumerate(days, start=1):
            if not isinstance(day, dict):
                issues.append({"code": "invalid_day", "message": f"day {day_index} is not an object"})
                continue
            if not str(day.get("date_label") or "").strip():
                issues.append({"code": "missing_date_label", "message": f"day {day_index} has no date_label"})

            slots = day.get("slots")
            if not isinstance(slots, list) or not slots:
                issues.append({"code": "missing_slots", "message": f"day {day_index} slots must not be empty"})
                continue

            for slot_index, slot in enumerate(slots, start=1):
                slot_count += 1
                if not isinstance(slot, dict):
                    issues.append({
                        "code": "invalid_slot",
                        "message": f"day {day_index} slot {slot_index} is not an object",
                    })
                    continue

                if not str(slot.get("title") or "").strip():
                    issues.append({
                        "code": "missing_slot_title",
                        "message": f"day {day_index} slot {slot_index} has no title",
                    })
                if slot.get("start_time") or slot.get("end_time"):
                    slots_with_time += 1
                if isinstance(slot.get("source_refs"), list) and slot.get("source_refs"):
                    slots_with_sources += 1
                if slot.get("activity") or slot.get("transport_to_next") or slot.get("ticket_info"):
                    slots_with_detail += 1

        if slot_count > 0 and slots_with_time == 0:
            issues.append({"code": "no_time_fields", "message": "no slot contains start_time or end_time"})
        if slot_count > 0 and slots_with_sources == 0 and not self._has_warning(plan):
            issues.append({
                "code": "no_sources_or_warning",
                "message": "slots have no source_refs and warnings do not explain evidence gaps",
            })
        if slot_count > 0 and slots_with_detail == 0:
            issues.append({
                "code": "missing_slot_details",
                "message": "slots should include activity, transport_to_next, or ticket_info",
            })

        return {
            "valid": len(issues) == 0,
            "issues": issues,
            "stats": {
                "day_count": len(days),
                "slot_count": slot_count,
                "slots_with_time": slots_with_time,
                "slots_with_sources": slots_with_sources,
                "slots_with_detail": slots_with_detail,
            },
        }

    @staticmethod
    def _has_warning(plan: dict[str, Any]) -> bool:
        warnings = plan.get("warnings")
        return isinstance(warnings, list) and any(str(item).strip() for item in warnings)


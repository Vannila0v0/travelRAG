import unittest

from agent_system.reporter.itinerary_validator import ItineraryValidator


class ItineraryValidatorTest(unittest.TestCase):
    def test_valid_itinerary_passes(self):
        plan = {
            "days": [
                {
                    "date_label": "第 1 天",
                    "slots": [
                        {
                            "start_time": "09:00",
                            "end_time": "11:00",
                            "title": "象鼻山",
                            "activity": "游览象鼻山",
                            "transport_to_next": "打车前往两江四湖",
                            "ticket_info": "以证据为准",
                            "source_refs": ["evidence_1"],
                        }
                    ],
                }
            ],
            "warnings": [],
        }

        result = ItineraryValidator().validate(plan)

        self.assertTrue(result["valid"])
        self.assertEqual(result["issues"], [])
        self.assertEqual(result["stats"]["day_count"], 1)
        self.assertEqual(result["stats"]["slot_count"], 1)
        self.assertEqual(result["stats"]["slots_with_sources"], 1)

    def test_missing_required_structure_fails(self):
        result = ItineraryValidator().validate({"days": []})

        self.assertFalse(result["valid"])
        self.assertIn("missing_days", {issue["code"] for issue in result["issues"]})

    def test_sources_can_be_replaced_by_warning(self):
        plan = {
            "days": [
                {
                    "date_label": "第 1 天",
                    "slots": [
                        {
                            "start_time": "09:00",
                            "title": "待确认景点",
                            "activity": "游览",
                            "source_refs": [],
                        }
                    ],
                }
            ],
            "warnings": ["证据不足，部分来源待确认"],
        }

        result = ItineraryValidator().validate(plan)

        self.assertTrue(
            all(issue["code"] != "no_sources_or_warning" for issue in result["issues"])
        )


if __name__ == "__main__":
    unittest.main()


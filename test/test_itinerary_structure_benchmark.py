import json
import unittest
from pathlib import Path

from agent_system.reporter.itinerary_validator import ItineraryValidator


DATASET = Path("evaluation/datasets/itinerary_structure_qa.jsonl")


def load_samples():
    samples = []
    with DATASET.open("r", encoding="utf-8") as file:
        for line in file:
            line = line.strip()
            if line:
                samples.append(json.loads(line))
    return samples


class ItineraryStructureBenchmarkTest(unittest.TestCase):
    def test_itinerary_structure_dataset(self):
        samples = load_samples()
        self.assertEqual(len(samples), 3)

        validator = ItineraryValidator()
        for sample in samples:
            result = validator.validate(sample["structured_output"])
            issue_codes = {issue["code"] for issue in result["issues"]}

            self.assertEqual(result["valid"], sample["expected_valid"], sample["id"])
            for code in sample.get("expected_issue_codes", []):
                self.assertIn(code, issue_codes, sample["id"])
            for code in sample.get("expected_absent_issue_codes", []):
                self.assertNotIn(code, issue_codes, sample["id"])


if __name__ == "__main__":
    unittest.main()


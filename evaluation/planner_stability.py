import argparse
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from agent_system.planner.task_decomposer import TaskDecomposer


DEFAULT_QUESTION = "帮我规划一天桂林市区游玩路线，包含交通和票价"
DEFAULT_REPORT_DIR = PROJECT_ROOT / "evaluation" / "reports"


def task_signature(task) -> dict[str, Any]:
    return {
        "task_id": task.task_id,
        "task_type": task.task_type,
        "depends_on": list(task.depends_on),
        "entities": list(task.entities),
        "description": task.description,
    }


def graph_signature(graph) -> list[dict[str, Any]]:
    return [task_signature(task) for task in graph.nodes]


def compact_signature(signature: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        {
            "task_id": item["task_id"],
            "task_type": item["task_type"],
            "depends_on": item["depends_on"],
            "entities": item["entities"],
        }
        for item in signature
    ]


def write_reports(report_dir: Path, report_name: str, payload: dict[str, Any]) -> None:
    report_dir.mkdir(parents=True, exist_ok=True)
    json_path = report_dir / f"{report_name}.json"
    md_path = report_dir / f"{report_name}.md"
    json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    summary = payload["summary"]
    lines = [
        "# Planner Stability Report",
        "",
        f"- Generated at: `{payload['generated_at']}`",
        f"- Question: {payload['question']}",
        f"- Runs: {summary['runs']}",
        f"- Stable: {summary['stable']}",
        f"- Unique signatures: {summary['unique_signatures']}",
        "",
        "## Runs",
        "",
    ]
    for run in payload["runs"]:
        lines.append(f"### Run {run['run']}")
        lines.append("")
        lines.append("| ID | Type | Depends On | Entities |")
        lines.append("| --- | --- | --- | --- |")
        for task in run["signature"]:
            lines.append(
                f"| {task['task_id']} | {task['task_type']} | "
                f"{', '.join(task['depends_on']) or '-'} | {', '.join(task['entities']) or '-'} |"
            )
        lines.append("")

    md_path.write_text("\n".join(lines), encoding="utf-8")


def evaluate(args: argparse.Namespace) -> dict[str, Any]:
    planner = TaskDecomposer()
    runs = []
    compact_signatures = []

    for index in range(1, args.runs + 1):
        graph = planner.decompose(args.question)
        signature = graph_signature(graph)
        compact = compact_signature(signature)
        compact_signatures.append(compact)
        runs.append(
            {
                "run": index,
                "execution_mode": graph.execution_mode,
                "signature": signature,
                "compact_signature": compact,
            }
        )

    unique = {
        json.dumps(signature, ensure_ascii=False, sort_keys=True)
        for signature in compact_signatures
    }
    payload = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "question": args.question,
        "summary": {
            "runs": args.runs,
            "stable": len(unique) == 1,
            "unique_signatures": len(unique),
        },
        "runs": runs,
    }
    write_reports(Path(args.report_dir), args.report_name, payload)
    return payload


def make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate planner DAG stability across repeated runs")
    parser.add_argument("--question", default=DEFAULT_QUESTION)
    parser.add_argument("--runs", type=int, default=3)
    parser.add_argument("--report-dir", default=str(DEFAULT_REPORT_DIR))
    parser.add_argument("--report-name", default="planner_stability")
    return parser


def main() -> int:
    payload = evaluate(make_parser().parse_args())
    summary = payload["summary"]
    print("\n=== Planner Stability Summary ===")
    print(f"runs: {summary['runs']}")
    print(f"stable: {summary['stable']}")
    print(f"unique_signatures: {summary['unique_signatures']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

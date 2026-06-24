import argparse
import json
import os
import statistics
import sys
import time
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Evaluation should be reproducible and should not stall on HuggingFace HEAD
# probes when the embedding model already exists in the local cache.
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

from query_engine import QueryEngine


DEFAULT_DATASET = PROJECT_ROOT / "evaluation" / "datasets" / "tourism_qa.jsonl"
DEFAULT_REPORT_DIR = PROJECT_ROOT / "evaluation" / "reports"


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as file:
        for line_no, line in enumerate(file, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSONL at {path}:{line_no}: {exc}") from exc
    return rows


def normalize_route(route: str | None) -> str:
    mapping = {
        "graph_local": "local",
        "graph_global": "global",
        "local_search": "local",
        "global_search": "global",
    }
    return mapping.get(route or "", route or "")


def contains_keyword(text: str, keyword: str) -> bool:
    return keyword.lower() in text.lower()


def source_text(source) -> str:
    values = [
        source.doc_id,
        source.chunk_id,
        source.source_path,
        source.file_name,
        getattr(source, "title", None),
        getattr(source, "url", None),
        getattr(source, "published_at", None),
        getattr(source, "source_type", None),
        source.text,
    ]
    return " ".join(str(value) for value in values if value)


def score_keywords(answer: str, expected_keywords: list[str]) -> dict[str, Any]:
    matched = [keyword for keyword in expected_keywords if contains_keyword(answer, keyword)]
    missing = [keyword for keyword in expected_keywords if keyword not in matched]
    total = len(expected_keywords)
    return {
        "recall": len(matched) / total if total else 1.0,
        "matched": matched,
        "missing": missing,
    }


def score_sources(sources, expected_sources: list[str]) -> dict[str, Any]:
    source_blob = "\n".join(source_text(source) for source in sources)
    matched = [keyword for keyword in expected_sources if contains_keyword(source_blob, keyword)]
    missing = [keyword for keyword in expected_sources if keyword not in matched]
    total = len(expected_sources)
    return {
        "recall": len(matched) / total if total else 1.0,
        "matched": matched,
        "missing": missing,
    }


def percentile(values: list[float], p: float) -> float:
    if not values:
        return 0.0
    if len(values) == 1:
        return values[0]
    ordered = sorted(values)
    index = (len(ordered) - 1) * p
    lower = int(index)
    upper = min(lower + 1, len(ordered) - 1)
    weight = index - lower
    return ordered[lower] * (1 - weight) + ordered[upper] * weight


def summarize_results(results: list[dict[str, Any]]) -> dict[str, Any]:
    evaluated = [item for item in results if not item.get("skipped")]
    latencies = [item["latency_seconds"] for item in evaluated]
    route_hits = [item["route_hit"] for item in evaluated]
    answer_recalls = [item["answer_keyword_recall"] for item in evaluated]
    source_recalls = [item["source_recall"] for item in evaluated]
    has_sources = [item["has_sources"] for item in evaluated]

    summary = {
        "total_samples": len(results),
        "evaluated_samples": len(evaluated),
        "skipped_samples": len(results) - len(evaluated),
        "route_accuracy": sum(route_hits) / len(route_hits) if route_hits else 0.0,
        "avg_answer_keyword_recall": statistics.mean(answer_recalls) if answer_recalls else 0.0,
        "avg_source_recall": statistics.mean(source_recalls) if source_recalls else 0.0,
        "has_source_rate": sum(has_sources) / len(has_sources) if has_sources else 0.0,
        "avg_latency_seconds": statistics.mean(latencies) if latencies else 0.0,
        "p50_latency_seconds": percentile(latencies, 0.50),
        "p95_latency_seconds": percentile(latencies, 0.95),
    }

    tag_buckets: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for item in evaluated:
        for tag in item.get("tags", []):
            tag_buckets[tag].append(item)

    tag_summary = {}
    for tag, rows in sorted(tag_buckets.items()):
        tag_summary[tag] = {
            "count": len(rows),
            "route_accuracy": sum(row["route_hit"] for row in rows) / len(rows),
            "avg_answer_keyword_recall": statistics.mean(row["answer_keyword_recall"] for row in rows),
            "avg_source_recall": statistics.mean(row["source_recall"] for row in rows),
            "avg_latency_seconds": statistics.mean(row["latency_seconds"] for row in rows),
        }
    summary["by_tag"] = tag_summary
    return summary


def failure_reasons(item: dict[str, Any]) -> list[str]:
    if item.get("skipped"):
        return ["skipped"]
    reasons = []
    if not item["route_hit"]:
        reasons.append("route_mismatch")
    if item["answer_keyword_recall"] < 0.6:
        reasons.append("low_answer_keyword_recall")
    if item["source_recall"] == 0:
        reasons.append("no_expected_source_hit")
    if not item["has_sources"]:
        reasons.append("no_sources")
    if item.get("error"):
        reasons.append("runtime_error")
    return reasons


def write_json_report(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def write_markdown_report(path: Path, payload: dict[str, Any]) -> None:
    summary = payload["summary"]
    results = payload["results"]
    failures = [item for item in results if item.get("failure_reasons")]

    lines = [
        "# Tourism QA Evaluation Report",
        "",
        f"- Generated at: `{payload['generated_at']}`",
        f"- Dataset: `{payload['dataset']}`",
        f"- Total samples: {summary['total_samples']}",
        f"- Evaluated samples: {summary['evaluated_samples']}",
        f"- Skipped samples: {summary['skipped_samples']}",
        "",
        "## Summary",
        "",
        "| Metric | Value |",
        "| --- | ---: |",
        f"| Route accuracy | {summary['route_accuracy']:.2%} |",
        f"| Avg answer keyword recall | {summary['avg_answer_keyword_recall']:.2%} |",
        f"| Avg source recall | {summary['avg_source_recall']:.2%} |",
        f"| Has source rate | {summary['has_source_rate']:.2%} |",
        f"| Avg latency seconds | {summary['avg_latency_seconds']:.2f} |",
        f"| P50 latency seconds | {summary['p50_latency_seconds']:.2f} |",
        f"| P95 latency seconds | {summary['p95_latency_seconds']:.2f} |",
        "",
        "## By Tag",
        "",
        "| Tag | Count | Route Acc | Answer Recall | Source Recall | Avg Latency |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]

    for tag, stats in summary["by_tag"].items():
        lines.append(
            f"| {tag} | {stats['count']} | {stats['route_accuracy']:.2%} | "
            f"{stats['avg_answer_keyword_recall']:.2%} | {stats['avg_source_recall']:.2%} | "
            f"{stats['avg_latency_seconds']:.2f} |"
        )

    lines.extend([
        "",
        "## Per Sample",
        "",
        "| ID | Expected | Actual | Route | Answer | Source | Latency |",
        "| --- | --- | --- | ---: | ---: | ---: | ---: |",
    ])
    for item in results:
        if item.get("skipped"):
            lines.append(
                f"| {item['id']} | {item['expected_route']} | skipped | - | - | - | - |"
            )
            continue
        lines.append(
            f"| {item['id']} | {item['expected_route']} | {item['actual_route']} | "
            f"{'yes' if item['route_hit'] else 'no'} | "
            f"{item['answer_keyword_recall']:.2%} | {item['source_recall']:.2%} | "
            f"{item['latency_seconds']:.2f} |"
        )

    lines.extend([
        "",
        "## Failure / Risk Samples",
        "",
    ])
    if not failures:
        lines.append("No failure or risk samples found.")
    else:
        for item in failures:
            lines.extend([
                f"### {item['id']}: {item['question']}",
                "",
                f"- Reasons: {', '.join(item['failure_reasons'])}",
                f"- Expected route: `{item.get('expected_route')}`",
                f"- Actual route: `{item.get('actual_route')}`",
                f"- Missing answer keywords: {item.get('missing_answer_keywords', [])}",
                f"- Missing sources: {item.get('missing_sources', [])}",
                "",
            ])

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def evaluate(args: argparse.Namespace) -> dict[str, Any]:
    if args.allow_hf_network:
        os.environ.pop("HF_HUB_OFFLINE", None)
        os.environ.pop("TRANSFORMERS_OFFLINE", None)

    dataset_path = Path(args.dataset).resolve()
    report_dir = Path(args.report_dir).resolve()
    samples = load_jsonl(dataset_path)
    if args.ids:
        wanted_ids = {item.strip() for item in args.ids.split(",") if item.strip()}
        samples = [sample for sample in samples if sample.get("id") in wanted_ids]
    if args.limit:
        samples = samples[: args.limit]

    engine = QueryEngine(index_dir=args.index_dir)
    results = []
    try:
        for index, sample in enumerate(samples, start=1):
            if args.skip_agent and sample.get("expected_route") == "agent":
                results.append(
                    {
                        "id": sample["id"],
                        "question": sample["question"],
                        "expected_route": sample.get("expected_route"),
                        "tags": sample.get("tags", []),
                        "skipped": True,
                        "skip_reason": "agent sample skipped by --skip-agent",
                    }
                )
                print(f"[{index}/{len(samples)}] {sample['id']} skipped agent")
                continue

            forced_route = args.route if args.route != "dataset" else sample.get("expected_route", "auto")
            if args.route == "auto":
                forced_route = "auto"

            print(f"[{index}/{len(samples)}] {sample['id']} route={forced_route} question={sample['question']}")
            start = time.perf_counter()
            error = None
            result = None
            try:
                sample_plan_mode = sample.get("plan_mode") or getattr(args, "plan_mode", "auto")
                result = engine.ask(
                    sample["question"],
                    route=forced_route,
                    report_mode=getattr(args, "report_mode", "concise"),
                    plan_mode=sample_plan_mode,
                    response_format=sample.get("response_format") or getattr(args, "response_format", "text"),
                )
            except Exception as exc:
                error = str(exc)
            latency = time.perf_counter() - start

            expected_route = normalize_route(sample.get("expected_route"))
            actual_route = normalize_route(result.route if result else None)
            answer = result.answer if result else ""
            sources = result.sources if result else []
            answer_score = score_keywords(answer, sample.get("expected_answer_keywords", []))
            source_score = score_sources(sources, sample.get("expected_sources", []))
            item = {
                "id": sample["id"],
                "question": sample["question"],
                "expected_route": expected_route,
                "actual_route": actual_route,
                "route_hit": actual_route == expected_route,
                "answer_keyword_recall": answer_score["recall"],
                "matched_answer_keywords": answer_score["matched"],
                "missing_answer_keywords": answer_score["missing"],
                "source_recall": source_score["recall"],
                "matched_sources": source_score["matched"],
                "missing_sources": source_score["missing"],
                "has_sources": bool(sources),
                "latency_seconds": latency,
                "answer": answer,
                "sources": [
                    {
                        "doc_id": source.doc_id,
                        "chunk_id": source.chunk_id,
                        "source_path": source.source_path,
                        "file_name": source.file_name,
                        "chunk_index": source.chunk_index,
                        "page": source.page,
                        "section": source.section,
                        "score": source.score,
                        "text": source.text if args.include_source_text else None,
                    }
                    for source in sources
                ],
                "metadata": result.metadata if result else {},
                "structured_output": result.structured_output if result else None,
                "plan_mode": sample.get("plan_mode") or getattr(args, "plan_mode", "auto"),
                "response_format": sample.get("response_format") or getattr(args, "response_format", "text"),
                "tags": sample.get("tags", []),
                "error": error,
            }
            item["failure_reasons"] = failure_reasons(item)
            results.append(item)
    finally:
        engine.close()

    payload = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "dataset": str(dataset_path),
        "route_mode": args.route,
        "report_mode": getattr(args, "report_mode", "concise"),
        "plan_mode": getattr(args, "plan_mode", "auto"),
        "response_format": getattr(args, "response_format", "text"),
        "skip_agent": args.skip_agent,
        "report_name": args.report_name,
        "summary": summarize_results(results),
        "results": results,
    }

    write_json_report(report_dir / "latest_eval.json", payload)
    write_markdown_report(report_dir / "latest_eval.md", payload)
    if args.report_name:
        write_json_report(report_dir / f"{args.report_name}.json", payload)
        write_markdown_report(report_dir / f"{args.report_name}.md", payload)
    return payload


def make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run rule-based evaluation for the tourism QueryEngine")
    parser.add_argument("--dataset", default=str(DEFAULT_DATASET), help="JSONL evaluation dataset")
    parser.add_argument("--report-dir", default=str(DEFAULT_REPORT_DIR), help="Report output directory")
    parser.add_argument("--report-name", help="Optional named report file, without extension")
    parser.add_argument("--index-dir", default=str(PROJECT_ROOT / ".cache" / "faiss_index"), help="FAISS index directory")
    parser.add_argument("--limit", type=int, help="Evaluate at most N samples")
    parser.add_argument("--ids", help="Comma-separated sample IDs to evaluate, for example q001,q012")
    parser.add_argument("--skip-agent", action="store_true", help="Skip samples whose expected route is agent")
    parser.add_argument(
        "--route",
        choices=["auto", "dataset", "vector", "local", "global", "hybrid", "agent"],
        default="auto",
        help="Route mode. auto uses router; dataset forces each sample's expected_route.",
    )
    parser.add_argument("--include-source-text", action="store_true", help="Store source text in JSON report")
    parser.add_argument(
        "--report-mode",
        choices=["concise", "full"],
        default="concise",
        help="Reporter mode for agent samples.",
    )
    parser.add_argument(
        "--plan-mode",
        choices=["auto", "detailed_itinerary", "place_recommendations"],
        default="auto",
        help="Planning mode for agent travel planning samples.",
    )
    parser.add_argument(
        "--response-format",
        choices=["text", "itinerary"],
        default="text",
        help="Response format for agent planning samples.",
    )
    parser.add_argument(
        "--allow-hf-network",
        action="store_true",
        help="Allow HuggingFace network checks/downloads during evaluation.",
    )
    return parser


def main() -> int:
    args = make_parser().parse_args()
    payload = evaluate(args)
    summary = payload["summary"]
    print("\n=== Evaluation Summary ===")
    print(f"evaluated_samples: {summary['evaluated_samples']}")
    print(f"route_accuracy: {summary['route_accuracy']:.2%}")
    print(f"avg_answer_keyword_recall: {summary['avg_answer_keyword_recall']:.2%}")
    print(f"avg_source_recall: {summary['avg_source_recall']:.2%}")
    print(f"has_source_rate: {summary['has_source_rate']:.2%}")
    print(f"avg_latency_seconds: {summary['avg_latency_seconds']:.2f}")
    print(f"reports: {Path(args.report_dir).resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

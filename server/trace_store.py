from __future__ import annotations

import json
from collections import Counter
from datetime import datetime
from pathlib import Path
from threading import Lock
from typing import Any


TRACE_DIR = Path(".cache") / "traces"
TRACE_FILE = TRACE_DIR / "query_traces.jsonl"
_TRACE_LOCK = Lock()


def append_trace(record: dict[str, Any]) -> None:
    TRACE_DIR.mkdir(parents=True, exist_ok=True)
    payload = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        **record,
    }
    with _TRACE_LOCK:
        with TRACE_FILE.open("a", encoding="utf-8") as file:
            file.write(json.dumps(payload, ensure_ascii=False) + "\n")


def list_traces(limit: int = 50) -> list[dict[str, Any]]:
    if not TRACE_FILE.exists():
        return []

    with _TRACE_LOCK:
        lines = TRACE_FILE.read_text(encoding="utf-8").splitlines()

    records = []
    for line in reversed(lines[-max(limit, 1):]):
        if not line.strip():
            continue
        try:
            records.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return records


def get_trace(trace_id: str) -> dict[str, Any] | None:
    if not TRACE_FILE.exists():
        return None

    with _TRACE_LOCK:
        lines = TRACE_FILE.read_text(encoding="utf-8").splitlines()

    for line in reversed(lines):
        if not line.strip():
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError:
            continue
        if record.get("trace_id") == trace_id:
            return record
    return None


def summarize_traces(limit: int = 200) -> dict[str, Any]:
    records = list_traces(limit=limit)
    if not records:
        return {
            "total": 0,
            "success": 0,
            "errors": 0,
            "degraded": 0,
            "avg_latency_ms": 0,
            "routes": {},
            "error_codes": {},
        }

    success_count = sum(1 for item in records if item.get("success") is True)
    degraded_count = sum(1 for item in records if item.get("degraded") is True)
    latencies = [
        item.get("latency_ms")
        for item in records
        if isinstance(item.get("latency_ms"), (int, float))
    ]

    route_counts = Counter(item.get("actual_route") or item.get("requested_route") or "unknown" for item in records)
    error_counts = Counter(item.get("error_code") for item in records if item.get("error_code"))

    return {
        "total": len(records),
        "success": success_count,
        "errors": len(records) - success_count,
        "degraded": degraded_count,
        "avg_latency_ms": round(sum(latencies) / len(latencies), 2) if latencies else 0,
        "routes": dict(route_counts),
        "error_codes": dict(error_counts),
    }

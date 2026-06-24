import html
import json
from typing import Any

from fastapi import APIRouter, HTTPException, Query, status
from fastapi.responses import HTMLResponse

from server.trace_store import get_trace, list_traces, summarize_traces


router = APIRouter(prefix="/dashboard", tags=["dashboard"])


def _escape(value: Any) -> str:
    if value is None:
        return ""
    return html.escape(str(value), quote=True)


def _format_json(value: Any) -> str:
    return html.escape(json.dumps(value, ensure_ascii=False, indent=2), quote=True)


def _render_distribution(items: dict[str, Any], empty_label: str) -> str:
    if not items:
        return f'<li class="muted">{_escape(empty_label)}</li>'
    return "\n".join(
        f"<li><span>{_escape(name)}</span><strong>{_escape(count)}</strong></li>"
        for name, count in sorted(items.items(), key=lambda item: str(item[0]))
    )


def _render_agent_trace(trace: dict[str, Any]) -> str:
    agent_trace = trace.get("agent_trace")
    if not isinstance(agent_trace, dict):
        return ""

    tasks = agent_trace.get("tasks")
    if not isinstance(tasks, list):
        tasks = []

    if tasks:
        task_rows = "\n".join(
            f"""
            <tr>
              <td>{_escape(task.get("task_id"))}</td>
              <td>{_escape(task.get("task_type"))}</td>
              <td>{_escape(task.get("status"))}</td>
              <td>{_escape(task.get("tool_name"))}</td>
              <td>{_escape(task.get("latency_ms"))}</td>
              <td>{_escape(task.get("source_count"))}</td>
              <td>{_escape(task.get("cache_hit"))}</td>
              <td>{_escape(task.get("error"))}</td>
            </tr>
            """
            for task in tasks
            if isinstance(task, dict)
        )
    else:
        task_rows = '<tr><td colspan="8" class="empty">No task-level trace recorded.</td></tr>'

    return f"""
    <div class="agent-trace">
      <div class="agent-metrics">
        <div><span>Planner</span><strong>{_escape(agent_trace.get("planner_latency_ms"))} ms</strong></div>
        <div><span>Execution</span><strong>{_escape(agent_trace.get("execution_latency_ms"))} ms</strong></div>
        <div><span>Reporter</span><strong>{_escape(agent_trace.get("reporter_latency_ms"))} ms</strong></div>
        <div><span>Mode</span><strong>{_escape(agent_trace.get("reporter_mode"))}</strong></div>
        <div><span>Plan</span><strong>{_escape(agent_trace.get("plan_mode"))}</strong></div>
        <div><span>LLM Calls</span><strong>{_escape(agent_trace.get("reporter_llm_calls"))}</strong></div>
        <div><span>Outline</span><strong>{_escape(agent_trace.get("outline_latency_ms"))} ms</strong></div>
        <div><span>Write</span><strong>{_escape(agent_trace.get("section_write_latency_ms"))} ms</strong></div>
        <div><span>Sections</span><strong>{_escape(agent_trace.get("section_count"))}</strong></div>
        <div><span>Tasks</span><strong>{_escape(agent_trace.get("task_count", len(tasks)))}</strong></div>
        <div><span>Cache</span><strong>{_escape(agent_trace.get("cache_hits", 0))} / {_escape(agent_trace.get("cache_misses", 0))}</strong></div>
      </div>
      <div class="task-table-wrap">
        <table class="task-table">
          <thead>
            <tr>
              <th>Task ID</th>
              <th>Type</th>
              <th>Status</th>
              <th>Tool</th>
              <th>Latency</th>
              <th>Sources</th>
              <th>Cache Hit</th>
              <th>Error</th>
            </tr>
          </thead>
          <tbody>{task_rows}</tbody>
        </table>
      </div>
    </div>
    """


def _render_itinerary(trace: dict[str, Any]) -> str:
    plan = trace.get("structured_output")
    if not isinstance(plan, dict):
        return ""

    days = plan.get("days")
    if not isinstance(days, list) or not days:
        return ""

    validation = trace.get("itinerary_validation")
    valid = validation.get("valid") if isinstance(validation, dict) else None
    issues = validation.get("issues") if isinstance(validation, dict) else []
    stats = validation.get("stats") if isinstance(validation, dict) else {}
    validity_class = "ok" if valid is True else "fail" if valid is False else "neutral"
    validity_label = "Valid" if valid is True else "Invalid" if valid is False else "Unchecked"

    issue_items = ""
    if isinstance(issues, list) and issues:
        issue_items = "\n".join(
            f"<li><strong>{_escape(issue.get('code'))}</strong> {_escape(issue.get('message'))}</li>"
            for issue in issues
            if isinstance(issue, dict)
        )
    else:
        issue_items = '<li class="muted">No validation issues.</li>'

    day_blocks = []
    for day in days:
        if not isinstance(day, dict):
            continue
        slots = day.get("slots")
        if not isinstance(slots, list):
            slots = []
        slot_items = []
        for slot in slots:
            if not isinstance(slot, dict):
                continue
            time_label = " - ".join(
                item for item in [str(slot.get("start_time") or ""), str(slot.get("end_time") or "")]
                if item
            ) or "Time TBD"
            details = [
                ("Activity", slot.get("activity")),
                ("Transport", slot.get("transport_to_next")),
                ("Ticket", slot.get("ticket_info")),
                ("Cost", slot.get("estimated_cost")),
                ("Notes", slot.get("notes")),
            ]
            detail_html = "\n".join(
                f"<p><span>{label}</span>{_escape(value)}</p>"
                for label, value in details
                if value
            )
            refs = slot.get("source_refs")
            refs_text = ", ".join(str(item) for item in refs) if isinstance(refs, list) else ""
            refs_html = f"<p><span>Sources</span>{_escape(refs_text)}</p>" if refs_text else ""
            slot_items.append(
                f"""
                <li>
                  <div class="slot-time">{_escape(time_label)}</div>
                  <div class="slot-body">
                    <strong>{_escape(slot.get("title") or "Untitled")}</strong>
                    <em>{_escape(slot.get("location"))}</em>
                    {detail_html}
                    {refs_html}
                  </div>
                </li>
                """
            )
        if not slot_items:
            slot_items.append('<li class="empty">No itinerary slots recorded.</li>')
        day_blocks.append(
            f"""
            <div class="itinerary-day">
              <h4>{_escape(day.get("date_label") or "Day")}</h4>
              <ol class="itinerary-slots">
                {''.join(slot_items)}
              </ol>
            </div>
            """
        )

    total_budget = plan.get("total_budget")
    assumptions = plan.get("assumptions") if isinstance(plan.get("assumptions"), list) else []
    warnings = plan.get("warnings") if isinstance(plan.get("warnings"), list) else []
    assumptions_html = "".join(f"<li>{_escape(item)}</li>" for item in assumptions) or '<li class="muted">None</li>'
    warnings_html = "".join(f"<li>{_escape(item)}</li>" for item in warnings) or '<li class="muted">None</li>'

    return f"""
    <div class="itinerary-preview">
      <div class="itinerary-head">
        <h3>Itinerary Preview</h3>
        <span class="badge {validity_class}">{validity_label}</span>
      </div>
      <div class="itinerary-stats">
        <span>Days: {_escape(stats.get("day_count", len(days)) if isinstance(stats, dict) else len(days))}</span>
        <span>Slots: {_escape(stats.get("slot_count", "")) if isinstance(stats, dict) else ""}</span>
        <span>With Sources: {_escape(stats.get("slots_with_sources", "")) if isinstance(stats, dict) else ""}</span>
      </div>
      {''.join(day_blocks)}
      <div class="itinerary-meta">
        <div><strong>Total Budget</strong><p>{_escape(total_budget or "Not specified")}</p></div>
        <div><strong>Assumptions</strong><ul>{assumptions_html}</ul></div>
        <div><strong>Warnings</strong><ul>{warnings_html}</ul></div>
        <div><strong>Validation Issues</strong><ul>{issue_items}</ul></div>
      </div>
    </div>
    """


def _render_trace_rows(traces: list[dict[str, Any]]) -> str:
    if not traces:
        return """
        <tr>
          <td colspan="7" class="empty">No trace records yet.</td>
        </tr>
        """

    rows = []
    for trace in traces:
        trace_id = _escape(trace.get("trace_id"))
        success = "OK" if trace.get("success") is True else "Failed"
        degraded = "Yes" if trace.get("degraded") is True else "No"
        success_class = "ok" if trace.get("success") is True else "fail"
        degraded_class = "warn" if trace.get("degraded") is True else "neutral"
        agent_trace_html = _render_agent_trace(trace)
        itinerary_html = _render_itinerary(trace)
        rows.append(
            f"""
            <tr>
              <td><a href="/dashboard/traces/{trace_id}">{trace_id}</a></td>
              <td>{_escape(trace.get("timestamp"))}</td>
              <td>{_escape(trace.get("requested_route"))}</td>
              <td>{_escape(trace.get("actual_route"))}</td>
              <td>{_escape(trace.get("latency_ms"))}</td>
              <td><span class="badge {success_class}">{success}</span></td>
              <td><span class="badge {degraded_class}">{degraded}</span></td>
            </tr>
            <tr class="trace-detail">
              <td colspan="7">
                <details>
                  <summary>{_escape(trace.get("question") or "Trace detail")}</summary>
                  {agent_trace_html}
                  {itinerary_html}
                  <pre>{_format_json(trace)}</pre>
                </details>
              </td>
            </tr>
            """
        )
    return "\n".join(rows)


def _render_dashboard(summary: dict[str, Any], traces: list[dict[str, Any]]) -> str:
    routes = summary.get("routes") if isinstance(summary.get("routes"), dict) else {}
    error_codes = summary.get("error_codes") if isinstance(summary.get("error_codes"), dict) else {}
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>GraphRAG Dashboard</title>
  <style>
    :root {{
      color-scheme: light;
      --bg: #f6f7f9;
      --panel: #ffffff;
      --text: #18212f;
      --muted: #667085;
      --line: #d8dee8;
      --accent: #0f766e;
      --accent-soft: #dff5f1;
      --danger: #b42318;
      --danger-soft: #fee4e2;
      --warn: #9a6700;
      --warn-soft: #fff2c6;
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      font-family: "Segoe UI", Arial, sans-serif;
      background: var(--bg);
      color: var(--text);
      letter-spacing: 0;
    }}
    header {{
      border-bottom: 1px solid var(--line);
      background: var(--panel);
    }}
    .wrap {{
      width: min(1180px, calc(100vw - 32px));
      margin: 0 auto;
    }}
    .topbar {{
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 16px;
      padding: 18px 0;
    }}
    h1 {{
      margin: 0;
      font-size: 24px;
      font-weight: 650;
    }}
    .links {{
      display: flex;
      gap: 8px;
      flex-wrap: wrap;
    }}
    .links a {{
      color: var(--accent);
      border: 1px solid var(--line);
      background: #fff;
      padding: 7px 10px;
      border-radius: 6px;
      text-decoration: none;
      font-size: 13px;
    }}
    main {{ padding: 20px 0 32px; }}
    .metrics {{
      display: grid;
      grid-template-columns: repeat(5, minmax(130px, 1fr));
      gap: 10px;
      margin-bottom: 16px;
    }}
    .metric {{
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 8px;
      padding: 14px;
      min-height: 92px;
    }}
    .metric span {{
      display: block;
      color: var(--muted);
      font-size: 13px;
      margin-bottom: 10px;
    }}
    .metric strong {{
      display: block;
      font-size: 28px;
      line-height: 1;
    }}
    .grid {{
      display: grid;
      grid-template-columns: repeat(2, minmax(0, 1fr));
      gap: 14px;
      margin-bottom: 16px;
    }}
    section {{
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 8px;
      overflow: hidden;
    }}
    section h2 {{
      margin: 0;
      padding: 13px 14px;
      border-bottom: 1px solid var(--line);
      font-size: 16px;
    }}
    ul.dist {{
      list-style: none;
      margin: 0;
      padding: 8px 14px 12px;
    }}
    ul.dist li {{
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 16px;
      padding: 8px 0;
      border-bottom: 1px solid #eef1f5;
      font-size: 14px;
    }}
    ul.dist li:last-child {{ border-bottom: 0; }}
    .muted, .empty {{ color: var(--muted); }}
    .table-wrap {{ overflow-x: auto; }}
    table {{
      width: 100%;
      border-collapse: collapse;
      table-layout: fixed;
      min-width: 860px;
    }}
    th, td {{
      padding: 10px 12px;
      border-bottom: 1px solid #eef1f5;
      text-align: left;
      vertical-align: top;
      font-size: 13px;
      overflow-wrap: anywhere;
    }}
    th {{
      color: var(--muted);
      font-weight: 600;
      background: #fbfcfe;
    }}
    td a {{ color: var(--accent); text-decoration: none; }}
    .trace-detail td {{
      padding-top: 0;
      background: #fbfcfe;
    }}
    details summary {{
      cursor: pointer;
      color: var(--muted);
      padding: 4px 0 8px;
    }}
    .agent-trace {{
      border: 1px solid var(--line);
      border-radius: 8px;
      background: #fff;
      margin: 0 0 10px;
      overflow: hidden;
    }}
    .agent-metrics {{
      display: grid;
      grid-template-columns: repeat(5, minmax(120px, 1fr));
      gap: 0;
      border-bottom: 1px solid var(--line);
    }}
    .agent-metrics div {{
      padding: 10px 12px;
      border-right: 1px solid #eef1f5;
      min-height: 62px;
    }}
    .agent-metrics div:last-child {{ border-right: 0; }}
    .agent-metrics span {{
      display: block;
      color: var(--muted);
      font-size: 12px;
      margin-bottom: 6px;
    }}
    .agent-metrics strong {{
      display: block;
      font-size: 16px;
    }}
    .task-table-wrap {{
      overflow-x: auto;
    }}
    .task-table {{
      min-width: 760px;
      table-layout: fixed;
    }}
    .task-table th,
    .task-table td {{
      font-size: 12px;
      padding: 8px 10px;
    }}
    .itinerary-preview {{
      border: 1px solid var(--line);
      border-radius: 8px;
      background: #fff;
      margin: 0 0 10px;
      padding: 12px;
    }}
    .itinerary-head {{
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 12px;
      margin-bottom: 8px;
    }}
    .itinerary-head h3 {{
      margin: 0;
      font-size: 15px;
    }}
    .itinerary-stats {{
      display: flex;
      flex-wrap: wrap;
      gap: 8px;
      margin-bottom: 12px;
      color: var(--muted);
      font-size: 12px;
    }}
    .itinerary-stats span {{
      border: 1px solid #eef1f5;
      border-radius: 6px;
      padding: 4px 7px;
      background: #fbfcfe;
    }}
    .itinerary-day {{
      border-top: 1px solid #eef1f5;
      padding-top: 10px;
      margin-top: 10px;
    }}
    .itinerary-day h4 {{
      margin: 0 0 8px;
      font-size: 14px;
    }}
    .itinerary-slots {{
      list-style: none;
      margin: 0;
      padding: 0;
    }}
    .itinerary-slots li {{
      display: grid;
      grid-template-columns: 112px minmax(0, 1fr);
      gap: 10px;
      padding: 9px 0;
      border-bottom: 1px solid #eef1f5;
    }}
    .itinerary-slots li:last-child {{ border-bottom: 0; }}
    .slot-time {{
      color: var(--accent);
      font-weight: 650;
      font-size: 12px;
      line-height: 1.5;
    }}
    .slot-body strong {{
      display: block;
      font-size: 14px;
      margin-bottom: 2px;
    }}
    .slot-body em {{
      display: block;
      color: var(--muted);
      font-style: normal;
      font-size: 12px;
      margin-bottom: 5px;
    }}
    .slot-body p {{
      margin: 3px 0;
      font-size: 12px;
      line-height: 1.45;
    }}
    .slot-body p span {{
      color: var(--muted);
      display: inline-block;
      min-width: 66px;
    }}
    .itinerary-meta {{
      display: grid;
      grid-template-columns: repeat(2, minmax(0, 1fr));
      gap: 10px;
      border-top: 1px solid #eef1f5;
      padding-top: 10px;
      margin-top: 10px;
      font-size: 12px;
    }}
    .itinerary-meta strong {{
      display: block;
      margin-bottom: 5px;
    }}
    .itinerary-meta p,
    .itinerary-meta ul {{
      margin: 0;
      padding-left: 16px;
      color: var(--muted);
    }}
    pre {{
      margin: 0 0 8px;
      padding: 12px;
      background: #101828;
      color: #eef4ff;
      border-radius: 6px;
      overflow-x: auto;
      font-size: 12px;
      line-height: 1.5;
    }}
    .badge {{
      display: inline-flex;
      align-items: center;
      justify-content: center;
      min-width: 54px;
      border-radius: 999px;
      padding: 3px 8px;
      font-size: 12px;
      font-weight: 600;
    }}
    .badge.ok {{ color: var(--accent); background: var(--accent-soft); }}
    .badge.fail {{ color: var(--danger); background: var(--danger-soft); }}
    .badge.warn {{ color: var(--warn); background: var(--warn-soft); }}
    .badge.neutral {{ color: var(--muted); background: #eef1f5; }}
    @media (max-width: 860px) {{
      .topbar {{ align-items: flex-start; flex-direction: column; }}
      .metrics {{ grid-template-columns: repeat(2, minmax(0, 1fr)); }}
      .grid {{ grid-template-columns: 1fr; }}
      .agent-metrics {{ grid-template-columns: repeat(2, minmax(0, 1fr)); }}
      .itinerary-meta {{ grid-template-columns: 1fr; }}
    }}
    @media (max-width: 520px) {{
      .wrap {{ width: min(100vw - 20px, 1180px); }}
      .metrics {{ grid-template-columns: 1fr; }}
      .agent-metrics {{ grid-template-columns: 1fr; }}
      .itinerary-slots li {{ grid-template-columns: 1fr; }}
      h1 {{ font-size: 21px; }}
    }}
  </style>
</head>
<body>
  <header>
    <div class="wrap topbar">
      <h1>GraphRAG Dashboard</h1>
      <nav class="links" aria-label="Dashboard API links">
        <a href="/dashboard/summary">Summary API</a>
        <a href="/dashboard/traces">Traces API</a>
      </nav>
    </div>
  </header>
  <main class="wrap">
    <div class="metrics">
      <div class="metric"><span>Total Requests</span><strong>{_escape(summary.get("total", 0))}</strong></div>
      <div class="metric"><span>Success</span><strong>{_escape(summary.get("success", 0))}</strong></div>
      <div class="metric"><span>Errors</span><strong>{_escape(summary.get("errors", 0))}</strong></div>
      <div class="metric"><span>Degraded</span><strong>{_escape(summary.get("degraded", 0))}</strong></div>
      <div class="metric"><span>Avg Latency</span><strong>{_escape(summary.get("avg_latency_ms", 0))} ms</strong></div>
    </div>
    <div class="grid">
      <section>
        <h2>Route Distribution</h2>
        <ul class="dist">
          {_render_distribution(routes, "No route data")}
        </ul>
      </section>
      <section>
        <h2>Error Codes</h2>
        <ul class="dist">
          {_render_distribution(error_codes, "No errors recorded")}
        </ul>
      </section>
    </div>
    <section>
      <h2>Recent Requests</h2>
      <div class="table-wrap">
        <table>
          <thead>
            <tr>
              <th style="width: 23%;">Trace ID</th>
              <th style="width: 18%;">Timestamp</th>
              <th style="width: 11%;">Requested</th>
              <th style="width: 11%;">Actual</th>
              <th style="width: 10%;">Latency</th>
              <th style="width: 10%;">Status</th>
              <th style="width: 10%;">Degraded</th>
            </tr>
          </thead>
          <tbody>
            {_render_trace_rows(traces)}
          </tbody>
        </table>
      </div>
    </section>
  </main>
</body>
</html>"""


@router.get("", response_class=HTMLResponse)
@router.get("/", response_class=HTMLResponse)
def dashboard_page(
    summary_limit: int = Query(default=200, ge=1, le=5000),
    trace_limit: int = Query(default=50, ge=1, le=500),
) -> HTMLResponse:
    summary = summarize_traces(limit=summary_limit)
    traces = list_traces(limit=trace_limit)
    return HTMLResponse(_render_dashboard(summary, traces))


@router.get("/summary")
def dashboard_summary(
    limit: int = Query(default=200, ge=1, le=5000),
) -> dict[str, Any]:
    return summarize_traces(limit=limit)


@router.get("/traces")
def dashboard_traces(
    limit: int = Query(default=50, ge=1, le=500),
) -> dict[str, Any]:
    traces = list_traces(limit=limit)
    return {
        "count": len(traces),
        "traces": traces,
    }


@router.get("/traces/{trace_id}")
def dashboard_trace_detail(trace_id: str) -> dict[str, Any]:
    trace = get_trace(trace_id)
    if trace is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Trace not found: {trace_id}",
        )
    return trace

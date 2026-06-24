import logging
import json
import re
import time
from typing import Any, Optional
from ..core.state import PlanExecuteState
from ..core.report_spec import ReportOutline, ReportResult, SectionContent, SectionOutline
from ..config.prompts.reporter_prompts import BATCH_SECTION_WRITE_PROMPT, CONCISE_REPORT_PROMPT
from ..integration.llm_factory import get_llm_model
from .outline_builder import OutlineBuilder
from .section_writer import SectionWriter

_LOGGER = logging.getLogger(__name__)


class BaseReporter:
    """报告生成器基类：编排大纲与写作流程"""

    def __init__(self, llm=None):
        self._llm = llm or get_llm_model()
        self.outline_builder = OutlineBuilder(llm=self._llm)
        self.section_writer = SectionWriter(llm=self._llm)
        self.last_llm_call_count = 0
        self.last_metrics: dict[str, int | str] = {}

    def generate(self, state: PlanExecuteState, mode: str = "concise") -> Optional[ReportResult]:
        """执行生成流程"""
        if not state.execution_records:
            _LOGGER.warning("没有执行记录，无法生成报告")
            return None

        if mode == "full":
            return self._generate_full(state)
        return self._generate_concise(state)

    def _generate_concise(self, state: PlanExecuteState) -> ReportResult:
        evidence_context = self._format_evidence(state)
        started = time.perf_counter()
        prompt = CONCISE_REPORT_PROMPT.format(
            query=state.input_query,
            evidence_context=evidence_context[:6000],
            plan_mode_instruction=self._plan_mode_instruction(getattr(state, "plan_mode", "auto")),
        )

        _LOGGER.info("📝 [Reporter] 正在生成简短汇总...")
        response = self._llm.invoke(prompt)
        final_report = str(response.content if hasattr(response, "content") else response)
        final_report = self._append_route_summary(final_report, state)
        self.last_llm_call_count = 1
        self.last_metrics = {
            "outline_latency_ms": 0,
            "section_write_latency_ms": int((time.perf_counter() - started) * 1000),
            "evidence_chars": len(evidence_context),
            "section_count": 1,
        }

        outline = ReportOutline(
            title="简短回答",
            report_type="concise",
            abstract=final_report[:200],
            sections=[
                SectionOutline(
                    section_id="summary",
                    title="回答",
                    description="基于多智能体检索证据生成的简短回答。",
                )
            ],
        )
        result = ReportResult(
            outline=outline,
            sections=[
                SectionContent(
                    section_id="summary",
                    title="回答",
                    content=final_report,
                    citations=[],
                )
            ],
            final_report=final_report,
        )
        state.final_report = final_report
        return result

    def _generate_full(self, state: PlanExecuteState) -> Optional[ReportResult]:
        # 1. 生成大纲
        evidence_context = self._format_evidence(state)
        outline_started = time.perf_counter()
        outline = self.outline_builder.build(state)
        outline_latency_ms = int((time.perf_counter() - outline_started) * 1000)
        llm_calls = 1
        if not outline.sections:
            self.last_llm_call_count = llm_calls
            self.last_metrics = {
                "outline_latency_ms": outline_latency_ms,
                "section_write_latency_ms": 0,
                "evidence_chars": len(evidence_context),
                "section_count": 0,
            }
            return None

        # 2. 批量写作所有章节，避免逐章多次 LLM 调用
        write_started = time.perf_counter()
        sections_content = self._write_sections_batch(
            outline,
            evidence_context,
            plan_mode=getattr(state, "plan_mode", "auto"),
        )
        section_write_latency_ms = int((time.perf_counter() - write_started) * 1000)
        llm_calls += 1

        full_text = [f"# {outline.title}\n", f"**摘要**: {outline.abstract}\n"]

        for sec_content in sections_content:
            full_text.append(f"## {sec_content.title}")
            full_text.append(sec_content.content)
            full_text.append("\n")

        final_report = "\n".join(full_text)
        final_report = self._append_route_summary(final_report, state)

        # 3. 封装结果
        result = ReportResult(
            outline=outline,
            sections=sections_content,
            final_report=final_report
        )

        # 4. 更新全局状态
        state.final_report = final_report
        self.last_llm_call_count = llm_calls
        self.last_metrics = {
            "outline_latency_ms": outline_latency_ms,
            "section_write_latency_ms": section_write_latency_ms,
            "evidence_chars": len(evidence_context),
            "section_count": len(sections_content),
        }
        return result

    def _write_sections_batch(
        self,
        outline: ReportOutline,
        evidence_context: str,
        plan_mode: str = "auto",
    ) -> list[SectionContent]:
        sections_payload = [
            {
                "section_id": section.section_id,
                "title": section.title,
                "description": section.description,
            }
            for section in outline.sections
        ]
        prompt = BATCH_SECTION_WRITE_PROMPT.format(
            title=outline.title,
            abstract=outline.abstract,
            sections_json=json.dumps(sections_payload, ensure_ascii=False, indent=2),
            evidence_context=evidence_context[:8000],
            plan_mode_instruction=self._plan_mode_instruction(plan_mode),
        )

        _LOGGER.info("✍️ [Reporter] 正在批量撰写 %s 个章节...", len(outline.sections))
        response = self._llm.invoke(prompt)
        content = str(response.content if hasattr(response, "content") else response)
        return self._parse_batch_sections(content, outline)

    def _parse_batch_sections(self, content: str, outline: ReportOutline) -> list[SectionContent]:
        parsed: dict | None = None
        try:
            match = re.search(r"\{.*\}", content, re.DOTALL)
            parsed = json.loads(match.group(0) if match else content)
        except Exception as exc:
            _LOGGER.error("批量章节解析失败: %s", exc)

        sections_by_id = {}
        if isinstance(parsed, dict) and isinstance(parsed.get("sections"), list):
            for item in parsed["sections"]:
                if not isinstance(item, dict):
                    continue
                section_id = str(item.get("section_id") or "").strip()
                if not section_id:
                    continue
                sections_by_id[section_id] = item

        fallback_content = content.strip()
        results = []
        for section in outline.sections:
            item = sections_by_id.get(section.section_id, {})
            results.append(
                SectionContent(
                    section_id=section.section_id,
                    title=str(item.get("title") or section.title),
                    content=str(item.get("content") or fallback_content or "该章节未能生成内容。"),
                    citations=[],
                )
            )
        return results

    def _format_evidence(self, state: PlanExecuteState) -> str:
        text = []
        for index, record in enumerate(state.execution_records, start=1):
            route = getattr(record, "route", None) or "unknown"
            output = self._format_record_output(record)
            text.append(f"[evidence_{index}] route={route}\n{output}")
        return "\n\n".join(text)

    def _format_record_output(self, record) -> str:
        route_summary = self._format_route_record(record)
        if route_summary:
            return route_summary
        return str(record.output)

    def _append_route_summary(self, final_report: str, state: PlanExecuteState) -> str:
        route_blocks = self._route_evidence_blocks(state)
        if not route_blocks or "路线交通依据" in final_report:
            return final_report
        return "\n\n".join(
            [
                final_report.strip(),
                "## 路线交通依据",
                "\n\n".join(route_blocks),
            ]
        ).strip()

    def _route_evidence_blocks(self, state: PlanExecuteState) -> list[str]:
        blocks = []
        for record in state.execution_records:
            block = self._format_route_record(record)
            if block:
                blocks.append(block)
        return blocks

    def _format_route_record(self, record) -> str | None:
        if getattr(record, "route", None) != "map_route":
            return None
        payload = self._parse_route_payload(getattr(record, "output", None))
        if not payload:
            return None

        route_order = [str(item) for item in payload.get("route_order") or [] if str(item).strip()]
        legs = payload.get("legs") if isinstance(payload.get("legs"), list) else []
        if not route_order and not legs:
            return None

        provider = payload.get("provider") or (getattr(record, "tool_metadata", {}) or {}).get("provider")
        mode = payload.get("mode")
        total_travel = self._optional_int(payload.get("total_travel_time_min"))
        total_distance = self._optional_int(payload.get("total_distance_m"))
        finish_time = payload.get("estimated_finish_time")
        feasible = payload.get("feasible")
        warnings = payload.get("warnings") if isinstance(payload.get("warnings"), list) else []

        lines = ["路线排序结果："]
        if route_order:
            lines.append(f"- 推荐顺序：{' -> '.join(route_order)}")

        summary_parts = []
        if total_travel is not None:
            summary_parts.append(f"约 {total_travel} 分钟")
        if total_distance is not None:
            summary_parts.append(self._format_distance(total_distance))
        if summary_parts:
            line = f"- 总交通：{'，'.join(summary_parts)}"
            if finish_time:
                line += f"；预计完成时间：{finish_time}"
            if feasible is not None:
                line += f"；可行性：{'可行' if feasible else '存在时间风险'}"
            lines.append(line)

        if provider or mode:
            lines.append(f"- 数据来源：{provider or 'unknown'}；交通方式：{mode or 'unknown'}")

        if legs:
            lines.append("- 路段耗时：")
            for index, leg in enumerate(legs, start=1):
                start = leg.get("from") or "上一站"
                end = leg.get("to") or "下一站"
                duration = self._optional_int(leg.get("duration_min"))
                distance = self._optional_int(leg.get("distance_m"))
                leg_parts = []
                if duration is not None:
                    leg_parts.append(f"约 {duration} 分钟")
                if distance is not None:
                    leg_parts.append(self._format_distance(distance))
                suffix = f"：{'，'.join(leg_parts)}" if leg_parts else ""
                lines.append(f"  {index}. {start} -> {end}{suffix}")

        if warnings:
            lines.append("- 路线风险：" + "；".join(str(item) for item in warnings if str(item).strip()))
        return "\n".join(lines)

    @staticmethod
    def _parse_route_payload(output: Any) -> dict[str, Any] | None:
        if isinstance(output, dict):
            return output
        if not isinstance(output, str):
            return None
        try:
            parsed = json.loads(output)
        except Exception:
            match = re.search(r"\{.*\}", output, re.DOTALL)
            if not match:
                return None
            try:
                parsed = json.loads(match.group(0))
            except Exception:
                return None
        return parsed if isinstance(parsed, dict) else None

    @staticmethod
    def _optional_int(value: Any) -> int | None:
        try:
            return int(value)
        except (TypeError, ValueError):
            return None

    @staticmethod
    def _format_distance(distance_m: int) -> str:
        if distance_m >= 1000:
            return f"约 {distance_m / 1000:.1f} 公里"
        return f"约 {distance_m} 米"

    @staticmethod
    def _plan_mode_instruction(plan_mode: str) -> str:
        if plan_mode == "detailed_itinerary":
            return (
                "用户选择了详细路线安排。答案必须按时间顺序组织，尽量包含上午/下午/晚上或按天安排、"
                "景点顺序、交通衔接、票价/预算和注意事项；不要只给景点清单。"
            )
        if plan_mode == "place_recommendations":
            return (
                "用户选择了景点/项目推荐。答案应先推荐值得玩的地方或项目，说明亮点、适合人群、"
                "大致游玩建议和注意事项；不要强行生成精确到小时的路线。"
            )
        return "用户未显式选择规划模式。按问题本身选择合适的组织方式。"

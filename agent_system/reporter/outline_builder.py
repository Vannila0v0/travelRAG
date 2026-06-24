import logging
import json
from typing import List
from ..core.state import PlanExecuteState
from ..core.report_spec import ReportOutline, SectionOutline
from ..config.prompts.reporter_prompts import OUTLINE_GEN_PROMPT
from ..integration.llm_factory import get_llm_model

_LOGGER = logging.getLogger(__name__)


class OutlineBuilder:
    """负责生成报告大纲"""

    def __init__(self, llm=None):
        self._llm = llm or get_llm_model()

    def build(self, state: PlanExecuteState) -> ReportOutline:
        """根据 State 生成大纲"""
        query = state.input_query

        # 简单汇总证据作为上下文
        evidence_summary = self._summarize_evidence(state.execution_records)

        prompt = OUTLINE_GEN_PROMPT.format(
            query=query,
            background="基于 GraphRAG 的多智能体检索结果",
            plan_mode_instruction=self._plan_mode_instruction(getattr(state, "plan_mode", "auto")),
            evidence_summary=evidence_summary[:2000]  # 截断防止超长
        )

        _LOGGER.info("📝 [Reporter] 正在构思大纲...")
        response = self._llm.invoke(prompt)
        content = str(response.content if hasattr(response, "content") else response)

        return self._parse_json(content)

    def _summarize_evidence(self, records: List) -> str:
        summary = []
        for i, record in enumerate(records):
            # 假设 record.output 是字符串或 dict
            content = str(record.output)[:100]
            summary.append(f"Evidence {i}: {content}")
        return "\n".join(summary)

    def _parse_json(self, content: str) -> ReportOutline:
        try:
            # 简单的 JSON 提取逻辑
            import re
            match = re.search(r"\{.*\}", content, re.DOTALL)
            if match:
                data = json.loads(match.group(0))
            else:
                data = json.loads(content)

            sections = [SectionOutline(**s) for s in data.get("sections", [])]
            return ReportOutline(
                title=data.get("title", "Analysis Report"),
                abstract=data.get("abstract", ""),
                sections=sections
            )
        except Exception as e:
            _LOGGER.error(f"大纲解析失败: {e}")
            # 兜底大纲
            return ReportOutline(
                title="报告生成失败",
                abstract="无法解析大纲",
                sections=[]
            )

    @staticmethod
    def _plan_mode_instruction(plan_mode: str) -> str:
        if plan_mode == "detailed_itinerary":
            return "输出应围绕详细路线安排组织，优先形成按天或按上午/下午/晚上推进的章节。"
        if plan_mode == "place_recommendations":
            return "输出应围绕景点/项目推荐组织，优先比较候选地点、亮点、适合人群和注意事项，不强行排成完整路线。"
        return "根据用户问题自然组织报告结构。"

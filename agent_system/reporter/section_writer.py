import logging
from typing import List, Dict
from ..core.report_spec import SectionContent, SectionOutline
from ..core.execution_record import ExecutionRecord
from ..config.prompts.reporter_prompts import SECTION_WRITE_PROMPT
from ..integration.llm_factory import get_llm_model

_LOGGER = logging.getLogger(__name__)


class SectionWriter:
    """负责撰写单个章节"""

    def __init__(self, llm=None):
        self._llm = llm or get_llm_model()

    def write_section(
        self,
        section: SectionOutline,
        records: List[ExecutionRecord],
        plan_mode: str = "auto",
    ) -> SectionContent:
        """撰写单章"""
        _LOGGER.info(f"✍️ [Reporter] 正在撰写章节: {section.title}")

        # 1. 准备证据上下文 (筛选与本章相关的证据)
        # 简化逻辑：这里暂时传入所有证据，实际项目中应该用语义检索筛选
        context = self._format_evidence(records)

        # 2. 构建 Prompt
        prompt = SECTION_WRITE_PROMPT.format(
            section_title=section.title,
            section_description=section.description,
            evidence_context=context,
            plan_mode_instruction=self._plan_mode_instruction(plan_mode),
        )

        # 3. LLM 生成
        response = self._llm.invoke(prompt)
        content = str(response.content if hasattr(response, "content") else response)

        return SectionContent(
            section_id=section.section_id,
            title=section.title,
            content=content,
            citations=[]  # 暂时未实现自动提取引用ID
        )

    def _format_evidence(self, records: List[ExecutionRecord]) -> str:
        text = []
        for i, rec in enumerate(records):
            # 给证据打上 ID 标签，方便引用
            evidence_id = f"evidence_{i + 1}"
            content = str(rec.output)
            text.append(f"[{evidence_id}]: {content}")
        return "\n\n".join(text)

    @staticmethod
    def _plan_mode_instruction(plan_mode: str) -> str:
        if plan_mode == "detailed_itinerary":
            return "本章节服务于详细路线安排，需关注时间顺序、交通、票价、预算和注意事项。"
        if plan_mode == "place_recommendations":
            return "本章节服务于景点/项目推荐，需关注候选地点亮点、适合人群、推荐理由和注意事项。"
        return "根据章节说明自然组织内容。"

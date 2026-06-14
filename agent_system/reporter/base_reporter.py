import logging
from typing import Optional
from ..core.state import PlanExecuteState
from ..core.report_spec import ReportResult
from .outline_builder import OutlineBuilder
from .section_writer import SectionWriter

_LOGGER = logging.getLogger(__name__)


class BaseReporter:
    """报告生成器基类：编排大纲与写作流程"""

    def __init__(self):
        self.outline_builder = OutlineBuilder()
        self.section_writer = SectionWriter()

    def generate(self, state: PlanExecuteState) -> Optional[ReportResult]:
        """执行生成流程"""
        if not state.execution_records:
            _LOGGER.warning("没有执行记录，无法生成报告")
            return None

        # 1. 生成大纲
        outline = self.outline_builder.build(state)
        if not outline.sections:
            return None

        # 2. 逐章写作
        sections_content = []
        full_text = [f"# {outline.title}\n", f"**摘要**: {outline.abstract}\n"]

        for sec_outline in outline.sections:
            # 调用 Writer
            sec_content = self.section_writer.write_section(
                sec_outline,
                state.execution_records
            )
            sections_content.append(sec_content)

            # 拼接到全文
            full_text.append(f"## {sec_content.title}")
            full_text.append(sec_content.content)
            full_text.append("\n")

        final_report = "\n".join(full_text)

        # 3. 封装结果
        result = ReportResult(
            outline=outline,
            sections=sections_content,
            final_report=final_report
        )

        # 4. 更新全局状态
        state.final_report = final_report
        return result
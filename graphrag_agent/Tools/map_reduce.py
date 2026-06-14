import json
import asyncio
import re
from pydantic import BaseModel, Field
from typing import List, Optional


# --- 数据结构 ---
class EvidenceSummary(BaseModel):
    """强制大模型输出的轻量级结构化对象"""
    key_points: List[str] = Field(default_factory=list, description="关键论点")
    entities: List[str] = Field(default_factory=list, description="涉及的实体")
    summary_text: str = Field(default="", description="精简摘要")
    token_count: int = Field(default=0, description="Token 估算值")


# --- 阶段一：Map 阶段 (并发提炼) ---
class EvidenceMapper:
    def __init__(self, llm, token_counter_func):
        self.llm = llm
        self.count_tokens = token_counter_func

    async def map_parallel(self, raw_documents: List[str], query: str) -> List[EvidenceSummary]:
        """开启多个并发线程，呼叫 LLM 提炼原始文档"""
        print(f"   [Mapper] 开启并发 Map 任务，共 {len(raw_documents)} 个文档块...")
        # 利用 asyncio.gather 并发执行
        tasks = [self._extract_single_doc(doc, query) for doc in raw_documents]
        results = await asyncio.gather(*tasks)
        # 过滤掉提取失败的空结果
        return [r for r in results if r is not None]

    async def _extract_single_doc(self, doc_text: str, query: str) -> Optional[EvidenceSummary]:
        """处理单个文档，强制输出 JSON"""
        prompt = f"""
        请阅读以下资料，并提取与用户问题相关的核心信息。
        要求必须返回纯 JSON 格式，不要任何 Markdown 标记或多余解释。

        【JSON 格式要求】:
        {{
            "key_points": ["核心论点1", "核心论点2"],
            "entities": ["实体A", "实体B"],
            "summary_text": "对这份资料的精简一句话总结"
        }}

        【用户问题】: {query}
        【参考资料】: {doc_text}
        """
        try:
            # 异步调用大模型
            response = await self.llm.ainvoke(prompt)
            content = response.content

            # 清洗大模型可能输出的 Markdown 代码块标签 (```json ... ```)
            content = re.sub(r"```json\s*", "", content)
            content = re.sub(r"```\s*", "", content)

            data = json.loads(content)

            summary = EvidenceSummary(
                key_points=data.get("key_points", []),
                entities=data.get("entities", []),
                summary_text=data.get("summary_text", ""),
                token_count=self.count_tokens(data.get("summary_text", ""))
            )
            return summary
        except Exception as e:
            print(f"   [Warn] Map 提取 JSON 失败: {e}")
            return None


# --- 阶段二：Reduce 阶段 (聚沙成塔) ---
class SectionReducer:
    def __init__(self, llm, token_counter_func, max_tokens=2000):
        self.llm = llm
        self.count_tokens = token_counter_func
        self.max_tokens = max_tokens  # 安全上下文阈值

    async def reduce(self, summaries: List[EvidenceSummary], query: str, strategy="tree") -> str:
        """主入口：根据策略选择归约方式"""
        if not summaries:
            return "（无有效参考资料）"

        print(f"   [Reducer] 启动 {strategy.upper()} 归约，处理 {len(summaries)} 个摘要...")
        if strategy == "tree":
            return await self._tree_reduce(summaries, query)
        else:
            return await self._collapse_reduce(summaries, query)

    # ---------------- 策略 A: Tree Reduce (树状归约) ----------------
    async def _tree_reduce(self, summaries: List[EvidenceSummary], query: str) -> str:
        """两两合并，自底向上构建二叉树"""
        # 递归终止条件
        total_tokens = sum(s.token_count for s in summaries)
        if len(summaries) == 1 or total_tokens < self.max_tokens:
            return await self._generate_final_text(summaries, query)

        next_level = []
        # 步长为 2 遍历，两两合并
        tasks = []
        for i in range(0, len(summaries), 2):
            if i + 1 < len(summaries):
                # 凑成一对，交给 LLM 融合
                tasks.append(self._merge_two_summaries(summaries[i], summaries[i + 1], query))
            else:
                # 落单的直接进入下一层
                next_level.append(summaries[i])

        # 并发执行这一层的两两合并
        merged_results = await asyncio.gather(*tasks)
        next_level.extend([r for r in merged_results if r is not None])

        # 递归升维
        return await self._tree_reduce(next_level, query)

    async def _merge_two_summaries(self, left: EvidenceSummary, right: EvidenceSummary, query: str) -> EvidenceSummary:
        """将两个摘要融合为一个更宏观的摘要"""
        prompt = f"""
        请将以下两份摘要信息合并为一份。剔除重复信息，保留核心论点和所有重要实体。
        【JSON 输出格式】必须与前面相同。
        【摘要A】: {left.model_dump_json()}
        【摘要B】: {right.model_dump_json()}
        """
        try:
            response = await self.llm.ainvoke(prompt)
            content = re.sub(r"```json\s*|```\s*", "", response.content)
            data = json.loads(content)
            return EvidenceSummary(
                key_points=data.get("key_points", []),
                entities=list(set(left.entities + right.entities)),  # 实体求并集
                summary_text=data.get("summary_text", ""),
                token_count=self.count_tokens(data.get("summary_text", ""))
            )
        except Exception:
            # 如果融合失败，退化为简单的文本拼接
            fallback_text = left.summary_text + "\n" + right.summary_text
            return EvidenceSummary(summary_text=fallback_text, token_count=self.count_tokens(fallback_text))

    # ---------------- 策略 B: Collapse Reduce (折叠归约) ----------------
    async def _collapse_reduce(self, summaries: List[EvidenceSummary], query: str) -> str:
        """贪心打包：能塞多少塞多少，快满了再压缩"""
        total_tokens = sum(s.token_count for s in summaries)
        # 容量检测：如果安全，直接生成
        if total_tokens < self.max_tokens:
            return await self._generate_final_text(summaries, query)

        intermediate = []
        current_batch = []
        current_tokens = 0

        # 贪心累加
        for summary in summaries:
            if current_tokens + summary.token_count > self.max_tokens:
                # 暂存区要爆炸了，触发批量压缩
                compressed_summary = await self._generate_intermediate_summary(current_batch)
                intermediate.append(compressed_summary)
                # 清空重置暂存区
                current_batch = [summary]
                current_tokens = summary.token_count
            else:
                current_batch.append(summary)
                current_tokens += summary.token_count

        # 处理最后一批残余
        if current_batch:
            compressed_summary = await self._generate_intermediate_summary(current_batch)
            intermediate.append(compressed_summary)

        # 递归处理中间结果
        return await self._collapse_reduce(intermediate, query)

    async def _generate_intermediate_summary(self, batch: List[EvidenceSummary]) -> EvidenceSummary:
        """把一大批摘要压缩成一个超级摘要"""
        combined_text = "\n".join([f"- {s.summary_text}" for s in batch])
        prompt = f"""
        请将以下多条信息浓缩为一段高度精炼的综合摘要。保留关键数据和实体。
        【需要压缩的信息】:
        {combined_text}
        【直接输出摘要文本，不要返回 JSON，不要多余解释】。
        """
        try:
            response = await self.llm.ainvoke(prompt)
            text = response.content.strip()
            return EvidenceSummary(summary_text=text, token_count=self.count_tokens(text))
        except Exception:
            return EvidenceSummary(summary_text=combined_text, token_count=self.count_tokens(combined_text))

    async def _generate_final_text(self, final_summaries: List[EvidenceSummary], query: str) -> str:
        """最后一步：将安全的摘要列表转化为最终的字符串 Context"""
        final_context = "\n".join([
            f"论点: {', '.join(s.key_points)}\n内容: {s.summary_text}"
            for s in final_summaries if s.summary_text
        ])
        return final_context
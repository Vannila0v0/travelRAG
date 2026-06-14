import logging
from langchain_core.tools import StructuredTool
from pydantic import BaseModel, Field
import asyncio

# 引入刚刚写的 Map-Reduce 引擎和模型工厂的 token 计数器
from graphrag_agent.Tools.map_reduce import EvidenceMapper, SectionReducer
from agent_system.integration.llm_factory import count_tokens

logger = logging.getLogger(__name__)


class GlobalSearchInput(BaseModel):
    query: str = Field(description="用户输入的原始搜索问题")


class GlobalSearchTool:
    def __init__(self, llm, neo4j_handler):
        self.llm = llm
        self.neo4j = neo4j_handler

        # 实例化 Map-Reduce 引擎
        self.mapper = EvidenceMapper(llm=self.llm, token_counter_func=count_tokens)
        self.reducer = SectionReducer(llm=self.llm, token_counter_func=count_tokens, max_tokens=1500)

    async def asearch(self, query: str) -> str:
        """
        【主逻辑 (异步)】：利用 Map-Reduce 架构处理全局搜索
        """
        logger.info(f"\n🌍 [Global Search] 启动 Map-Reduce 宏观检索: {query}")

        # 1. 抓取所有摘要 (你原先的抓取逻辑)
        raw_documents = []
        with self.neo4j.driver.session() as session:
            # 取出所有有 summary 的社区（工业界可加上 LIMIT 限制总数比如 LIMIT 30）
            res = session.run("""
                MATCH (c:Community) 
                WHERE c.summary IS NOT NULL 
                RETURN c.level as lvl, c.title as title, c.summary as summary
                ORDER BY c.level DESC LIMIT 20
            """)
            for r in res:
                # 把每个社区拼成一段文本，作为一条 raw document
                raw_documents.append(f"社区层级:{r['lvl']} | 标题:{r['title']} | 内容:{r['summary']}")

        if not raw_documents:
            return "<global_context>\n（无相关宏观社区记录）\n</global_context>"

        # ==========================================
        # 👑 核心环节：执行 Map-Reduce 管道
        # ==========================================
        # 2. Map 阶段：并发将长文本提取为 JSON 对象
        logger.info(f"   -> [Map] 正在对 {len(raw_documents)} 篇图谱报告进行并发特征提取...")
        mapped_summaries = await self.mapper.map_parallel(raw_documents, query)

        # 3. Reduce 阶段：使用树状归约 (Tree Reduce) 合并信息
        logger.info(f"   -> [Reduce] 正在使用 Tree Reduce 策略降维合并信息...")
        final_reduced_context = await self.reducer.reduce(mapped_summaries, query, strategy="tree")
        # ==========================================

        logger.info(f"   -> [完成] 宏观检索完毕，最终精炼长度: {len(final_reduced_context)} 字符")

        # 4. 封装返回
        assembled_context = f"""
<global_context>
以下是利用 Map-Reduce 架构对全图谱社区报告进行归纳后的宏观分析：
{final_reduced_context}
</global_context>
"""
        return assembled_context

    def search(self, query: str) -> str:
        """同步包装器，兼容不支持异步的环境"""
        # 注意：在 LangGraph 中通常会直接调用 asearch
        try:
            loop = asyncio.get_event_loop()
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)

        if loop.is_running():
            # 如果已有事件循环，创建任务（通常在 LangGraph stream 内部会走到这里）
            import nest_asyncio
            nest_asyncio.apply()
            return loop.run_until_complete(self.asearch(query))
        else:
            return loop.run_until_complete(self.asearch(query))

    def get_tool(self) -> StructuredTool:
        """
        暴露为标准的 LangChain Tool，显式提供同步 (func) 和异步 (coroutine) 入口
        """
        return StructuredTool.from_function(
            func=self.search,
            coroutine=self.asearch,  # LangGraph 会优先使用异步版本，加速并发
            name="global_macro_search",
            description="当用户询问宏观趋势、整体总结、跨领域分析等需要阅读海量报告的问题时，必须调用此工具。",
            args_schema=GlobalSearchInput
        )
import logging

logger = logging.getLogger(__name__)


class GlobalSearch:
    def __init__(self, llm, neo4j_handler):
        self.llm = llm
        self.neo4j = neo4j_handler

    def search(self, query: str):
        logger.info(f"🔎 [Global Search] 启动递归检索模式: {query}")

        # 1. 抓取所有摘要，按层级分组
        # 我们优先看高层级（Level 1/2），因为它们是高度概括的
        summaries = {}
        with self.neo4j.driver.session() as session:
            # 按层级降序排列 (Level 2 -> Level 1 -> Level 0)
            res = session.run("""
                MATCH (c:Community) 
                WHERE c.summary IS NOT NULL 
                RETURN c.level as lvl, c.title as title, c.summary as summary
                ORDER BY c.level DESC
            """)
            for r in res:
                lvl = r['lvl']
                if lvl not in summaries: summaries[lvl] = []
                summaries[lvl].append(f"【{r['title']}】: {r['summary']}")

        # 2. 构建 "滚雪球" 上下文
        # 微软 GraphRAG 的 Map-Reduce 在这里可以简化为：
        # 先给 LLM 看 Level 1 (宏观)，再给看 Level 0 (微观)，让它自己融合。

        context_parts = []
        for lvl in sorted(summaries.keys(), reverse=True):
            section = f"\n=== 层级 {lvl} (Level {lvl} 宏观视角) ===\n" + "\n".join(summaries[lvl])
            context_parts.append(section)

        full_context = "\n".join(context_parts)

        # 3. 生成回答
        prompt = f"""
        你是一个拥有上帝视角的旅游数据分析师。请回答用户问题。

        【参考资料说明】
        资料采用“层级化”结构：
        - Level 高的段落是宏观总结（由下级汇总而来）。
        - Level 0 的段落是具体细节。

        【知识库】
        {full_context}

        【用户问题】
        {query}

        请生成一篇结构清晰的报告，既要有宏观结论，也要引用具体的微观细节作为支撑。
        """

        try:
            response = self.llm.invoke(prompt)

            # 兼容性判断：如果是对象则取 content，如果是字符串则直接使用
            if hasattr(response, 'content'):
                return response.content
            else:
                return str(response)

        except Exception as e:
            logger.error(f"LLM 生成失败: {e}")
            return "抱歉，生成回答时出现错误。"
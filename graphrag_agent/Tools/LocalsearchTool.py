import json
from langchain_core.tools import StructuredTool
from pydantic import BaseModel, Field


# 定义工具的输入格式，强制大模型按此格式调用
class SearchInput(BaseModel):
    query: str = Field(description="用户输入的原始搜索问题")


class LocalSearchTool:
    def __init__(self, llm, neo4j_handler, vector_search_func):
        """
        :param llm: 用于提取实体的 LLM (来自 llm_factory 的 get_llm_model)
        :param neo4j_handler: 你的 Neo4jHandler 实例
        :param vector_search_func: 你的向量检索函数
        """
        self.llm = llm
        self.neo4j = neo4j_handler
        self.vector_search = vector_search_func

    def extract_keywords(self, query: str) -> dict:
        """
        [适配 BaseAgent] 提取关键词，返回字典格式供缓存系统使用
        """
        prompt = f"""
        请从用户的问题中提取核心实体（如景点名、票种、地名等）。
        要求：
        1. 返回 Python 列表格式字符串。
        2. 不要多余的解释。
        用户问题：{query}
        输出示例：['夜游两江四湖', '成人票']
        """
        entities = []
        try:
            res = self.llm.invoke(prompt).content
            start = res.find('[')
            end = res.find(']') + 1
            if start != -1 and end != -1:
                import ast
                entities = ast.literal_eval(res[start:end])
        except Exception as e:
            print(f"   [Warn] LLM 实体提取失败，降级使用 Jieba: {e}")
            import jieba.posseg as pseg
            words = pseg.cut(query)
            entities = [w.word for w in words if w.flag.startswith('n')]

        # 组装为 BaseAgent 期望的字典格式
        return {
            "low_level": entities,  # 具体实体
            "high_level": []  # 宏观概念（在此工具中可留空）
        }

    def search(self, query: str) -> str:
        """
        【职责重构】: 只负责检索和拼接 Context，不负责生成最终回答！
        最终的生成交由 LangGraph 的 generate 节点处理。
        """
        print(f"\n🔎 [Local Search Tool] 开始检索: {query}")

        # 1. 实体识别
        keywords_dict = self.extract_keywords(query)
        entities = keywords_dict["low_level"]
        print(f"   -> 提取实体: {entities}")

        # 2. 图谱上下文检索
        graph_context = self.neo4j.get_local_context(entities) if hasattr(self.neo4j,
                                                                          'get_local_context') else "（无相关图谱记录）"
        if not graph_context:
            graph_context = "（无相关图谱记录）"
        print(f"   -> 图谱上下文长度: {len(str(graph_context))} 字符")

        # 3. 向量文档检索
        docs = self.vector_search(query)
        # 假设 docs 是 Document 对象列表
        text_context = "\n".join(
            [f"---文档片段---\n{d.page_content if hasattr(d, 'page_content') else str(d)}" for d in docs])
        print(f"   -> 向量上下文片段数: {len(docs) if docs else 0}")

        # 4. 上下文组装 (返回给 Agent 的 ToolMessage)
        assembled_context = f"""
<graph_context>
{graph_context}
</graph_context>

<vector_context>
{text_context}
</vector_context>
"""
        return assembled_context

    def get_tool(self) -> StructuredTool:
        """
        【核心适配】: 将普通方法包装为 LangChain Tool，供 Agent 绑定
        """
        return StructuredTool.from_function(
            func=self.search,
            name="local_hybrid_search",
            description="当用户询问具体的旅游景点、票价、路线等细节事实信息时，必须调用此工具获取图谱和文档资料。",
            args_schema=SearchInput
        )
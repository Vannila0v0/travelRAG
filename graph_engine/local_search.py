import jieba
import json
from langchain_core.prompts import PromptTemplate


class LocalSearch:
    def __init__(self, llm, neo4j_handler, vector_search_func):
        """
        :param llm: 你的 OllamaLLM 实例
        :param neo4j_handler: 你的 Neo4jHandler 实例
        :param vector_search_func: 一个函数，输入 query，返回 list[Document]
        """
        self.llm = llm
        self.neo4j = neo4j_handler
        self.vector_search = vector_search_func

    def _extract_entities(self, query: str):
        """
        [Step 1] 实体提取
        模仿 graph-rag-agent 的 extraction 逻辑，从问题中提取关键实体。
        """
        prompt = f"""
        请从用户的问题中提取核心实体（如景点名、票种、地名等）。

        要求：
        1. 返回 Python 列表格式字符串。
        2. 不要多余的解释。

        用户问题：{query}
        输出示例：['夜游两江四湖', '成人票']
        """
        try:
            res = self.llm.invoke(prompt)
            res = res.content if hasattr(res, "content") else str(res)
            # 简单的清洗，提取 [] 部分
            start = res.find('[')
            end = res.find(']') + 1
            if start != -1 and end != -1:
                # 安全起见，用 json 解析，或者 eval
                import ast
                return ast.literal_eval(res[start:end])
            return []
        except:
            # 兜底策略：使用 jieba 提取名词
            print("   [Warn] LLM 实体提取失败，降级使用 Jieba")
            import jieba.posseg as pseg
            words = pseg.cut(query)
            return [w.word for w in words if w.flag.startswith('n')]

    def search(self, query: str):
        print(f"\n🔎 [Local Search] 开始处理: {query}")

        # 1. 实体识别 (Extraction)
        entities = self._extract_entities(query)
        print(f"   -> 提取实体: {entities}")

        # 2. 图谱上下文检索 (Graph Retrieval)
        # 这里就是利用 GraphRAG 的优势：查找实体的直接关系网
        graph_context = self.neo4j.get_local_context(entities)
        if not graph_context:
            graph_context = "（无相关图谱记录）"
        print(f"   -> 图谱上下文长度: {len(graph_context)} 字符")

        # 3. 向量文档检索 (Vector Retrieval)
        # 利用你原本强大的混合检索逻辑
        docs = self.vector_search(query)
        text_context = "\n".join([f"---文档片段---\n{d.page_content}" for d in docs])
        print(f"   -> 向量上下文片段数: {len(docs)}")

        # 4. 上下文组装与生成 (Synthesis)
        # 构造一个结构化的 Prompt，强制让 LLM 结合两者
        system_prompt = f"""
        你是一个智能旅游助手。请综合利用下面的【知识图谱】和【参考文档】来回答用户问题。

        注意：
        1. 【知识图谱】中的信息（如价格、路线）通常比文档更准确，请优先参考。
        2. 如果图谱和文档都找不到答案，请诚实回答不知道。

        ### 知识图谱 (结构化数据):
        {graph_context}

        ### 参考文档 (详细文本):
        {text_context}

        ### 用户问题:
        {query}
        """

        response = self.llm.invoke(system_prompt)
        return response.content if hasattr(response, "content") else str(response)

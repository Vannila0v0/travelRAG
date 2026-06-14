from typing import List, Dict
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

# 引入我们写好的基类和两大搜索引擎
from graphrag_agent.agents.base import BaseAgent
from graphrag_agent.Tools.LocalsearchTool import LocalSearchTool
from graphrag_agent.Tools.GlobalsearchTool import GlobalSearchTool

# --- 需要补充到顶部的 import ---
from langchain_core.messages import AIMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

# 引入你的 Prompt 配置（请确保你的项目中有这些变量）
from graphrag_agent.config.prompts.hybrid_prompts import LC_SYSTEM_PROMPT, HYBRID_AGENT_GENERATE_PROMPT
from graphrag_agent.config.settings import response_type
# -----------------------------

class HybridAgent(BaseAgent):
    """使用混合搜索的Agent实现 (仿照原项目完美复刻)"""

    def __init__(self, neo4j_handler, vector_search_func):
        # 1. 挂载外部依赖
        self.neo4j_handler = neo4j_handler
        self.vector_search_func = vector_search_func

        # 2. 指定专属的 cache_dir
        self.cache_dir = "./cache/hybrid_agent"

        # 3. 调用父类构造函数 (会自动触发 _setup_tools 和图编译)
        super().__init__(cache_dir=self.cache_dir)

    def _setup_tools(self) -> List:
        """
        原项目使用 self.search_tool.get_tool() 和 get_global_tool()。
        由于我们拆分了 Local 和 Global，这里直接挂载两者的 tool。
        """
        self.local_tool_instance = LocalSearchTool(self.llm, self.neo4j_handler, self.vector_search_func)
        self.global_tool_instance = GlobalSearchTool(self.llm, self.neo4j_handler)

        return [
            self.local_tool_instance.get_tool(),
            self.global_tool_instance.get_tool()
        ]

    def _add_retrieval_edges(self, workflow):
        """仿照原项目：添加从检索到生成的边"""
        # 简单的从检索直接到生成
        workflow.add_edge("retrieve", "generate")

    def _extract_keywords(self, query: str) -> Dict[str, List[str]]:
        """
        仿照原项目：提取查询关键词，并加入独立的关键词缓存逻辑
        """
        # 1. 检查缓存中是否已经提取过该问题的关键词
        cache_key = f"keywords:{query}"
        cached_keywords = self.cache_manager.get(cache_key)
        if cached_keywords:
            return cached_keywords

        try:
            # 2. 分别调用我们的两个工具提取底层和高层关键词
            low_level = self.local_tool_instance.extract_keywords(query).get("low_level", [])
            high_level = self.global_tool_instance.extract_high_level_keywords(query)

            keywords = {
                "low_level": low_level,
                "high_level": high_level
            }

            # 3. 缓存结果，防止大模型重复浪费 Token 提取同一句话的关键词
            self.cache_manager.set(cache_key, keywords)

            return keywords
        except Exception as e:
            print(f"关键词提取失败: {e}")
            # 出错时返回默认空关键词
            return {"low_level": [], "high_level": []}

    def _generate_node(self, state):
        """
        生成回答节点逻辑
        职责：提取上下文 -> 校验双重缓存 -> 调用大模型生成 -> 回写双重缓存
        """
        messages = state["messages"]

        # 1. 安全地提取用户问题与检索到的文档上下文
        # LangGraph 的标准流转中，倒数第三条通常是原问题(Human)，倒数第一条是检索结果(Tool)
        try:
            question = messages[-3].content if len(messages) >= 3 else "未找到问题"
        except Exception:
            question = "无法获取问题"

        try:
            docs = messages[-1].content if messages[-1] else "未找到相关信息"
        except Exception:
            docs = "无法获取检索结果"

        # ==========================================
        # 👑 核心：双重缓存校验机制 (基于你的 CacheManager)
        # ==========================================
        # 获取当前会话ID，用于上下文感知缓存 (ContextAwareCacheStrategy)
        thread_id = state.get("configurable", {}).get("thread_id", "default")

        # 校验一层：全局缓存 (Global Cache)
        # 调用 manager.py 的 get()，不传 thread_id，默认使用 SimpleCacheStrategy
        global_result = self.global_cache_manager.get(question)
        if global_result:
            self._log_execution("generate",
                                {"question": question, "docs_length": len(docs)},
                                "全局缓存命中")
            return {"messages": [AIMessage(content=global_result)]}

        # 校验二层：会话缓存 (Session Cache)
        # 传入 thread_id，触发 manager.py 里的 ContextAwareCacheStrategy
        cached_result = self.cache_manager.get(question, thread_id=thread_id)
        if cached_result:
            self._log_execution("generate",
                                {"question": question, "docs_length": len(docs)},
                                "会话缓存命中")
            # 同步机制：会话缓存命中时，顺手将其同步到全局缓存，造福其他会话
            self.global_cache_manager.set(question, cached_result)
            return {"messages": [AIMessage(content=cached_result)]}

        # ==========================================
        # 🧠 大模型生成逻辑
        # ==========================================
        # 组装原项目的 Prompt 模板
        prompt = ChatPromptTemplate.from_messages([
            ("system", LC_SYSTEM_PROMPT),
            ("human", HYBRID_AGENT_GENERATE_PROMPT),
        ])

        # 构建标准的 LangChain LCEL 执行链
        rag_chain = prompt | self.llm | StrOutputParser()

        try:
            # 执行推理
            response = rag_chain.invoke({
                "context": docs,
                "question": question,
                "response_type": response_type
            })

            # 结果回写双重缓存
            if response and len(response) > 10:
                # 写入会话缓存 (带 thread_id)
                self.cache_manager.set(question, response, thread_id=thread_id)
                # 写入全局缓存 (不带 thread_id)
                self.global_cache_manager.set(question, response)

            self._log_execution("generate",
                                {"question": question, "docs_length": len(docs)},
                                response)

            return {"messages": [AIMessage(content=response)]}

        except Exception as e:
            error_msg = f"生成回答时出错: {str(e)}"
            self._log_execution("generate_error",
                                {"question": question, "docs_length": len(docs)},
                                error_msg)
            return {"messages": [AIMessage(content=f"抱歉，我无法回答这个问题。技术原因: {str(e)}")]}
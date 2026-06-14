from typing import Annotated, Sequence, TypedDict, List, Dict, Any, AsyncGenerator, Optional
from abc import ABC, abstractmethod
import time
import asyncio
import os

from langchain_core.messages import BaseMessage, HumanMessage
from langgraph.graph import END, StateGraph, START
from langgraph.prebuilt import ToolNode, tools_condition
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph.message import add_messages

# ==========================================
# [完美接入] 从你刚刚测试成功的工厂类中引入模型
# ==========================================
from agent_system.integration.llm_factory import (
    get_llm_model,
    get_stream_llm_model,
    get_embeddings_model
)

# [适配] 引入你的多级缓存系统
from graphrag_agent.cache_manager.manager import CacheManager
from graphrag_agent.cache_manager.config import CacheConfig
from graphrag_agent.config.settings import AGENT_SETTINGS


class BaseAgent(ABC):
    """Agent 基类，定义通用功能、LangGraph流转和缓存接口"""

    def __init__(self, cache_dir="./cache", memory_only=False):
        """
        初始化 Agent
        参数:
            cache_dir: 缓存目录
            memory_only: 是否仅使用内存缓存
        """
        # 1. 无缝加载本地大模型与向量模型（单例，极速加载）
        self.llm = get_llm_model()
        self.stream_llm = get_stream_llm_model()
        self.embeddings = get_embeddings_model()

        # 2. 加载基础配置
        self.default_recursion_limit = AGENT_SETTINGS.get("default_recursion_limit", 25)
        self.stream_flush_threshold = AGENT_SETTINGS.get("stream_flush_threshold", 10)
        self.chunk_size = AGENT_SETTINGS.get("chunk_size", 5)

        self.memory = MemorySaver()
        self.execution_log = []
        self.performance_metrics = {}

        # 3. 初始化多级缓存：会话缓存 (Session Cache)
        session_cache_config = CacheConfig(
            base_dir=cache_dir,
            backend_type='memory' if memory_only else 'hybrid',
            max_memory_items=200,
            enable_vector_match=True
        )
        self.cache_manager = CacheManager(config=session_cache_config)

        # 4. 初始化多级缓存：全局缓存 (Global Cache)
        global_cache_dir = os.path.join(cache_dir, "global")
        global_cache_config = CacheConfig(
            base_dir=global_cache_dir,
            backend_type='memory' if memory_only else 'hybrid',
            max_memory_items=500,
            enable_vector_match=True
        )
        self.global_cache_manager = CacheManager(config=global_cache_config)

        # 5. 初始化工具与工作流图 (交由具体业务子类实现)
        self.tools = self._setup_tools()
        self._setup_graph()

    @abstractmethod
    def _setup_tools(self) -> List:
        """设置该 Agent 可用的工具（子类必须实现）"""
        pass

    @abstractmethod
    def _add_retrieval_edges(self, workflow):
        """添加从检索(retrieve)到生成(generate)的路由边（子类必须实现）"""
        pass

    @abstractmethod
    def _extract_keywords(self, query: str) -> Dict[str, List[str]]:
        """提取查询关键词（用于缓存精准匹配）"""
        pass

    @abstractmethod
    def _generate_node(self, state):
        """生成最终回答的节点逻辑（子类必须实现）"""
        pass

    def _setup_graph(self):
        """搭建 LangGraph 核心状态机"""

        # 定义图状态 (利用 add_messages 自动追加消息)
        class AgentState(TypedDict):
            messages: Annotated[Sequence[BaseMessage], add_messages]

        workflow = StateGraph(AgentState)

        # 添加基础节点
        workflow.add_node("agent", self._agent_node)
        workflow.add_node("retrieve", ToolNode(self.tools))
        workflow.add_node("generate", self._generate_node)

        # 添加基础边
        workflow.add_edge(START, "agent")
        # 路由器：让大模型决定是直接结束，还是调用工具
        workflow.add_conditional_edges(
            "agent",
            tools_condition,
            {
                "tools": "retrieve",
                END: END,
            },
        )

        # 留给子类的核心业务扩展点：检索完后怎么走？
        self._add_retrieval_edges(workflow)
        workflow.add_edge("generate", END)

        # 编译工作流图
        self.graph = workflow.compile(checkpointer=self.memory)

    def _agent_node(self, state):
        """Agent 大脑节点：负责思考和决定是否调用工具"""
        messages = state["messages"]

        # 绑定工具调用大模型思考
        model = self.llm.bind_tools(self.tools)
        response = model.invoke(messages)

        self._log_execution("agent", messages, response)

        # 严格遵守 LangGraph 规范，返回增量更新
        return {"messages": [response]}

    # --- 缓存查询与管理 ---
    def _check_all_caches(self, query: str, thread_id: str = "default"):
        """依次检查全局缓存和会话缓存"""
        # 1. 查全局缓存
        global_result = self.global_cache_manager.get(query)
        if global_result:
            return global_result

        # 2. 查当前会话缓存
        session_result = self.cache_manager.get(query, thread_id=thread_id)
        if session_result:
            # 如果会话里有，顺手同步给全局
            self.global_cache_manager.set(query, session_result)
            return session_result

        return None

    # --- 对外交互接口 ---
    def ask(self, query: str, thread_id: str = "default", recursion_limit: Optional[int] = None):
        """同步阻塞式提问接口"""
        cached = self._check_all_caches(query, thread_id)
        if cached:
            return cached

        config = {
            "configurable": {
                "thread_id": thread_id,
                "recursion_limit": recursion_limit or self.default_recursion_limit
            }
        }

        inputs = {"messages": [HumanMessage(content=query)]}
        try:
            # 运行图
            for _ in self.graph.stream(inputs, config=config):
                pass

            # 提取最后的结果
            chat_history = self.memory.get(config)["channel_values"]["messages"]
            answer = chat_history[-1].content

            # 写入缓存
            if answer and len(answer) > 10:
                self.cache_manager.set(query, answer, thread_id=thread_id)
                self.global_cache_manager.set(query, answer)

            return answer
        except Exception as e:
            return f"执行出错: {str(e)}"

    async def ask_stream(self, query: str, thread_id: str = "default", recursion_limit: Optional[int] = None) -> \
    AsyncGenerator[str, None]:
        """异步流式提问接口 (原生 Graph astream 驱动，支持状态透出)"""

        # 1. 缓存拦截
        cached = self._check_all_caches(query, thread_id)
        if cached:
            # 命中缓存时，为了良好的前端体验，模拟打字机断句输出
            import re
            chunks = re.split(r'([.!?。！？]\s*)', cached)
            for chunk in chunks:
                if chunk:
                    yield chunk
                    await asyncio.sleep(0.02)
            return

        config = {
            "configurable": {
                "thread_id": thread_id,
                "recursion_limit": recursion_limit or self.default_recursion_limit
            }
        }
        inputs = {"messages": [HumanMessage(content=query)]}
        answer_buffer = ""

        try:
            # 2. 驱动 LangGraph 异步流转
            async for event in self.graph.astream(inputs, config=config, stream_mode="updates"):
                for node_name, node_data in event.items():
                    # 状态透出：向前端实时推送执行进度
                    if node_name == "agent":
                        yield "🤔 **Agent正在分析问题**...\n\n"
                    elif node_name == "retrieve":
                        yield "🔍 **正在检索本地知识图谱与向量库**...\n\n"
                    elif node_name == "generate":
                        # 捕捉最终生成节点的回答
                        final_message = node_data.get("messages", [])[-1]
                        content = final_message.content

                        # 按标点符号断句平滑输出
                        import re
                        chunks = re.split(r'([.!?。！？]\s*)', content)
                        for chunk in chunks:
                            if chunk:
                                yield chunk
                                answer_buffer += chunk
                                await asyncio.sleep(0.02)

            # 3. 写入双重缓存
            if answer_buffer and len(answer_buffer) > 10:
                self.cache_manager.set(query, answer_buffer, thread_id=thread_id)
                self.global_cache_manager.set(query, answer_buffer)

        except Exception as e:
            yield f"\n\n❌ **执行错误**: {str(e)}"

    def _log_execution(self, node_name: str, input_data: Any, output_data: Any):
        """记录节点执行日志"""
        self.execution_log.append({
            "node": node_name,
            "timestamp": time.time(),
            "input": str(input_data)[:500],
            "output": str(output_data)[:500]
        })

    def close(self):
        """关闭资源"""
        pass
import logging
import json
import re
from typing import Dict, Any, List
from ..core.plan_spec import TaskGraph, TaskNode
from ..config.prompts.planner_prompts import TASK_DECOMPOSE_PROMPT
from ..integration.llm_factory import get_llm_model
from .task_normalizer import TaskNormalizer

_LOGGER = logging.getLogger(__name__)


class TaskDecomposer:
    """任务分解器：将 Query -> TaskGraph"""

    def __init__(self):
        self._llm = get_llm_model()
        self._max_tasks = 5
        self._normalizer = TaskNormalizer()

    def decompose(self, query: str) -> TaskGraph:
        """执行分解逻辑"""
        # 1. 构造 Prompt
        prompt = TASK_DECOMPOSE_PROMPT.format(
            query=query,
            max_tasks=self._max_tasks
        )

        _LOGGER.info(f"🧠 [Planner] 正在思考任务拆解...")

        # 2. 调用 LLM
        response = self._llm.invoke(prompt)
        # 兼容不同版本的 LangChain 返回值
        content = response.content if hasattr(response, 'content') else str(response)

        # [调试关键] 打印 LLM 原始回复，方便排查 JSON 格式问题
        print(f"\n[DEBUG] Planner Raw Output:\n{content}\n")

        # 3. 解析 JSON
        parsed_json = self._parse_json_from_markdown(content)

        # 4. 构建并验证 TaskGraph
        task_graph = self._build_task_graph(parsed_json)
        return self._normalizer.normalize(query, task_graph)

    def _parse_json_from_markdown(self, text: str) -> Dict[str, Any]:
        """从 LLM 的 Markdown 输出中提取 JSON"""
        try:
            # 尝试提取 ```json ... ``` 块
            match = re.search(r"```(?:json)?\s*(.*?)```", text, re.DOTALL)
            if match:
                json_str = match.group(1).strip()
            else:
                # 假如没有代码块，尝试寻找第一个 { 和最后一个 }
                start = text.find('{')
                end = text.rfind('}')
                if start != -1 and end != -1:
                    json_str = text[start:end + 1]
                else:
                    json_str = text  # 实在没办法，硬解析全文

            return json.loads(json_str)
        except Exception as e:
            _LOGGER.error(f"JSON 解析失败: {e}")
            _LOGGER.error(f"问题文本片段: {text[:100]}...")
            # 返回一个空的结构，让 build_task_graph 处理兜底
            return {}

    def _build_task_graph(self, data: Dict[str, Any]) -> TaskGraph:
        """将字典转换为 Pydantic 模型"""
        nodes_data = data.get("nodes", [])
        clean_nodes = []

        # 如果解析失败导致 nodes 为空，尝试生成一个默认任务
        if not nodes_data:
            _LOGGER.warning("⚠️ 未能解析出有效节点，生成默认搜索任务")
            # 这里的默认任务可以让流程不至于直接断掉
            clean_nodes.append(TaskNode(
                task_type="local_search",
                description="搜索用户查询的相关信息",
                depends_on=[]
            ))
        else:
            for raw in nodes_data:
                # 简单清洗和默认值填充
                if "task_type" not in raw: raw["task_type"] = "local_search"
                if "status" not in raw: raw["status"] = "pending"
                if "depends_on" not in raw: raw["depends_on"] = []

                # 转换为 TaskNode 对象
                try:
                    node = TaskNode(**raw)
                    clean_nodes.append(node)
                except Exception as e:
                    _LOGGER.warning(f"跳过无效节点: {raw}, 错误: {e}")

        return TaskGraph(
            nodes=clean_nodes,
            execution_mode=data.get("execution_mode", "sequential")
        )

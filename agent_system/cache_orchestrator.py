import logging
import time
import sys
import os
from typing import Optional, Dict, Any

from graphrag_agent.cache_manager.manager import CacheManager
from graphrag_agent.cache_manager.config import CacheConfig


# 引入原有的核心组件 (保持你的引用不变)
from .core.state import PlanExecuteState
from .core.plan_spec import PlanSpec
from .planner.task_decomposer import TaskDecomposer
from .executor.worker_coordinator import WorkerCoordinator
from .reporter.base_reporter import BaseReporter

# 配置日志
logging.basicConfig(level=logging.INFO)
_LOGGER = logging.getLogger(__name__)


class MultiAgentOrchestrator:
    """
    集成了多级缓存系统的多智能体编排器。
    """

    def __init__(self,
                 enable_cache: bool = True,
                 cache_difficulty: str = 'hard',
                 model_name: str = "deepseek-chat"):
        """
        Args:
            enable_cache: 是否启用缓存
            cache_difficulty: 缓存策略等级 ('simple', 'hard')
            model_name: 用于生成缓存Key的上下文标识（不传给子组件，仅用于缓存区分）
        """
        # 1. 初始化原有核心组件 (不传参，遵照你的原代码)
        self.planner = TaskDecomposer()
        self.worker = WorkerCoordinator()
        self.reporter = BaseReporter()

        # 保存模型名称用于缓存上下文区分
        self.model_name = model_name

        # 2. 初始化缓存管理器
        self.enable_cache = enable_cache
        self.cache_manager = None

        if self.enable_cache:
            try:
                _LOGGER.info(f"正在初始化缓存系统 (策略: {cache_difficulty})...")
                cache_config = CacheConfig.for_agent(cache_difficulty)
                self.cache_manager = CacheManager(cache_config)
            except Exception as e:
                _LOGGER.error(f"缓存初始化失败，将以无缓存模式运行: {e}")
                self.enable_cache = False

    def run(self, query: str) -> PlanExecuteState:
        # 1. 初始化状态
        state = PlanExecuteState(
            session_id="session_1",
            input_query=query
        )
        _LOGGER.info(f"🚀 [Orchestrator] 开始执行: {query}")

        start_time = time.time()

        # === Phase 0: Cache Check (缓存检查) ===
        if self.enable_cache:
            # 构建上下文，确保不同模型的回答不混用
            cache_context = {
                "model": self.model_name,
                "role": "orchestrator"
            }

            cached_result = self.cache_manager.get(query, **cache_context)

            if cached_result:
                elapsed = time.time() - start_time
                _LOGGER.info(f"🚀 [Cache Hit] 发现缓存！耗时 {elapsed:.2f}s")
                _LOGGER.info(f"   - 内容预览: {str(cached_result)}...")

                # 直接填充结果并返回状态，跳过规划和执行
                state.final_report = cached_result
                # 可以选择标记 state 说明是来自缓存
                # state.metadata["from_cache"] = True
                return state

        # === Phase 1: Planning (规划) ===
        _LOGGER.info("--- Phase 1: Planning ---")

        try:
            # [关键修复] 使用 decompose 而不是 plan
            task_graph = self.planner.decompose(state.input_query)
        except Exception as e:
            _LOGGER.error(f"❌ Planner 运行出错: {e}")
            state.final_report = f"Planning Error: {str(e)}"
            return state

        # 将 TaskGraph 包装成 PlanSpec 并存入 state
        state.plan = PlanSpec(
            original_query=state.input_query,
            task_graph=task_graph
        )

        # 检查是否成功生成节点
        if not state.plan or not state.plan.task_graph.nodes:
            _LOGGER.error(f"❌ 规划失败，未生成任务。Plan状态: {state.plan}")
            state.final_report = "Planning failed: No tasks generated."
            return state

        _LOGGER.info(f"✅ 规划成功，生成 {len(state.plan.task_graph.nodes)} 个任务")
        for node in state.plan.task_graph.nodes:
            _LOGGER.info(f"   - [{node.task_type}] {node.description}")

        # === Phase 2: Execution (执行) ===
        _LOGGER.info("--- Phase 2: Execution ---")
        try:
            self.worker.run(state)
        except Exception as e:
            _LOGGER.error(f"❌ Worker 运行出错: {e}")
            state.final_report = f"Execution Error: {str(e)}"
            return state

        # === Phase 3: Reporting (报告) ===
        _LOGGER.info("--- Phase 3: Reporting ---")
        try:
            # 假设 generate 方法会更新 state.final_report
            self.reporter.generate(state)
        except Exception as e:
            _LOGGER.error(f"❌ Reporter 运行出错: {e}")
            if not state.final_report:
                state.final_report = f"Reporting Error: {str(e)}"

        # === Phase 4: Cache Update (缓存写入) ===
        if (self.enable_cache and
                state.final_report and
                not str(state.final_report).startswith("Error")):
            _LOGGER.info("💾 正在写入缓存...")
            self.cache_manager.set(query, state.final_report, **cache_context)

        total_time = time.time() - start_time
        _LOGGER.info(f"✅ 流程结束 (总耗时: {total_time:.2f}s)")
        return state

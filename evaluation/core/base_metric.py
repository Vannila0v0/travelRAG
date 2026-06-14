import re
from abc import ABC, abstractmethod
from typing import Dict, List, Tuple


class BaseMetric(ABC):
    """所有评估指标的基类"""

    # 指标名称，所有继承此类的子类必须重写这个名称
    metric_name = "base"

    def __init__(self, config):
        """
        初始化评估指标基类

        Args:
            config: 评估配置 (可以是字典或 EvaluatorConfig 对象)
        """
        # 支持字典或 EvaluatorConfig 对象
        if isinstance(config, dict):
            # 注意：这里需要你后续实现 evaluatorConfig.py
            from evaluation.evaluator_config.evaluatorConfig import EvaluatorConfig
            self.config = EvaluatorConfig(config)
        else:
            self.config = config

        self.dataset_name = self.config.get('dataset_name', 'default')
        self.debug = self.config.get('debug', False)
        # 获取传入的 LLM 模型实例，用于复杂的语义打分或规则失效时的回退评估
        self.llm = self.config.get('llm', None)

    @abstractmethod
    def calculate_metric(self, data) -> Tuple[Dict[str, float], List]:
        """
        计算评估指标（抽象方法，强制子类实现）

        Args:
            data: 评估数据对象 (如 AnswerEvaluationData 或 RetrievalEvaluationData)

        Returns:
            Tuple[Dict, List]: (总体评估结果字典, 每个样本的具体评分列表)
        """
        return {}, []

    def log(self, message, *args, **kwargs):
        """
        输出调试日志，只有在 debug 模式下才打印
        """
        if self.debug:
            print(f"[{self.__class__.__name__}] {message}", *args, **kwargs)

    def get_llm_fallback_score(self, prompt: str, default_score: float = 0.5) -> float:
        """
        核心亮点：使用 LLM 进行回退评分（LLM as a Judge）
        当传统的正则或规则匹配无法准确评估时（例如语义相近但表述不同），调用此方法让大模型打分。

        Args:
            prompt: 组装好的打分提示词
            default_score: 默认分数，当 LLM 评分失败、断网或输出格式不对时返回

        Returns:
            float: 提取出的 0.0 到 1.0 之间的分数
        """
        # 如果没有传入 LLM 实例，直接返回默认分数
        if not self.llm:
            self.log(f"  LLM不可用，使用默认分数: {default_score:.4f}")
            return default_score

        try:
            self.log("  正在使用LLM进行回退评分...")
            # 调用你项目中的 LLM 生成回复 (注意适配你自己的 LLM invoke 方法)
            response = self.llm.invoke(prompt)
            score_text = response.content if hasattr(response, 'content') else str(response)

            self.log(f"  LLM响应: {score_text}")

            # 使用正则提取回答中的第一个浮点数或整数
            score_match = re.search(r'(\d+(\.\d+)?)', score_text)
            if score_match:
                extracted_score = float(score_match.group(1))
                # 强行截断，确保分数在 0-1 范围内
                score = max(0.0, min(1.0, extracted_score))
                self.log(f"  LLM评分结果: {score:.4f}")
                return score
            else:
                self.log(f"  无法从LLM响应中提取分数，使用默认分数: {default_score:.4f}")
                return default_score
        except Exception as e:
            self.log(f"  LLM评分出错: {e}")
            return default_score
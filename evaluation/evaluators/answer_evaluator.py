import os
import traceback
from typing import Dict

# 导入你刚刚搭建好的基类和数据结构
from evaluation.core.base_evaluator import BaseEvaluator
from evaluation.core.evaluation_data import AnswerEvaluationData


class AnswerEvaluator(BaseEvaluator):
    """答案评估器，用于评估系统回答的质量"""

    def __init__(self, config):
        """
        初始化答案评估器

        Args:
            config: 评估配置 (字典或 EvaluatorConfig 实例)
        """
        super().__init__(config)

    def evaluate(self, data: AnswerEvaluationData) -> Dict[str, float]:
        """
        执行评估流程：遍历配置好的指标，逐一打分，并将分数回填到数据样本中。
        """
        self.log("\n======== 开始评估答案质量 ========")
        self.log(f"样本总数: {len(data.samples)}")
        self.log(f"使用的评估指标: {', '.join(self.metrics)}")

        result_dict = {}

        # 遍历配置中要求评估的所有指标 (例如: ["em", "f1"])
        for metric_name in self.metrics:
            try:
                self.log(f"\n开始计算指标: {metric_name}")
                # 获取具体的指标类名称用于日志打印 (如 ExactMatch, F1Score)
                metric_class_name = self.metric_class[metric_name].__class__.__name__
                self.log(f"使用评估类: {metric_class_name}")

                # 核心步骤：调用具体指标计算类的 calculate_metric 方法进行打分
                metric_result, metric_scores = self.metric_class[metric_name].calculate_metric(data)
                result_dict.update(metric_result)

                # 统计基本信息 - 兼容处理（有些复杂的深度评估指标可能会返回一个字典作为明细）
                if metric_scores and not isinstance(metric_scores[0], dict):
                    min_score = min(metric_scores)
                    max_score = max(metric_scores)
                    avg_score = sum(metric_scores) / len(metric_scores)
                    self.log(f"指标统计: 最小值={min_score:.4f}, 最大值={max_score:.4f}, 平均值={avg_score:.4f}")

                # 更新（登记成绩）每个样本的评分到 evaluation_data 容器中
                for sample, metric_score in zip(data.samples, metric_scores):
                    sample.update_evaluation_score(metric_name, metric_score)

                # 安全获取总体得分进行打印
                if metric_result:
                    overall_score = list(metric_result.values())[0]
                    self.log(f"完成指标 {metric_name} 计算，总体得分: {overall_score:.4f}")

            except Exception as e:
                # 某一个指标算崩了，不要影响其他指标的计算
                self.log(f'评估 {metric_name} 时出错: {e}')
                self.log(traceback.format_exc())
                continue

        self.log("\n所有指标计算结果:")
        for metric, score in result_dict.items():
            self.log(f"  {metric}: {score:.4f}")

        self.log("======== 答案质量评估结束 ========\n")

        # 将平均分结果保存到 txt 文本中 (依赖 BaseEvaluator 中的方法)
        if self.save_metric_flag:
            self.save_metric_score(result_dict)
            self.log(f"评估结果已保存至: {os.path.join(self.save_dir, 'metric_score.txt')}")

        # 将带有详细分数、多智能体日志、缓存信息的完整数据保存到 json (依赖 BaseEvaluator 中的方法)
        if self.save_data_flag:
            self.save_data(data)
            self.log(f"评估中间数据已保存至: {os.path.join(self.save_dir, 'intermediate_data.json')}")

        return result_dict
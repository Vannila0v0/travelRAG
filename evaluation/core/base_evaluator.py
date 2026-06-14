import os
import json
from abc import ABC, abstractmethod
from typing import Dict, Type

# 导入上面的 BaseMetric
from evaluation.core.base_metric import BaseMetric


class BaseEvaluator(ABC):
    """评估器基类，定义通用评估调度流程和文件处理接口"""

    def __init__(self, config):
        """
        初始化评估器
        """
        if isinstance(config, dict):
            # 注意：依赖你后续创建的 EvaluatorConfig
            from evaluation.evaluator_config.evaluatorConfig import EvaluatorConfig
            self.config = EvaluatorConfig(config)
        else:
            self.config = config

        self.save_dir = self.config.get('save_dir', './evaluation_results')
        self.save_metric_flag = self.config.get('save_metric_score', True)
        self.save_data_flag = self.config.get('save_intermediate_data', True)
        self.metrics = self.config.get_metrics()
        self.debug = self.config.get('debug', False)

        # 确保保存评估结果的目录存在
        os.makedirs(self.save_dir, exist_ok=True)

        # 黑魔法：自动获取所有继承了 BaseMetric 的可用评估指标
        self.available_metrics = self._collect_metrics()

        # 根据配置中的需求，实例化需要的评估指标
        self.metric_class = {}
        for metric in self.metrics:
            if metric in self.available_metrics:
                # 实例化指标类，并将大管家的 config 传递给它
                self.metric_class[metric] = self.available_metrics[metric](self.config.to_dict())
            else:
                print(f"{metric} 评估指标未实现!")
                raise NotImplementedError(f"评估指标 {metric} 未在代码中找到对应的类")

    def _collect_metrics(self) -> Dict[str, Type[BaseMetric]]:
        """
        核心设计：利用 Python 反射机制（__subclasses__）收集所有评估指标。
        这样你以后新增任何 Metric 类，都不需要手动注册，系统会自动发现它。
        """

        def find_descendants(base_class, subclasses=None):
            if subclasses is None:
                subclasses = set()

            direct_subclasses = base_class.__subclasses__()
            for subclass in direct_subclasses:
                if subclass not in subclasses:
                    subclasses.add(subclass)
                    # 递归查找，防止有指标类继承了另一个指标类
                    find_descendants(subclass, subclasses)
            return subclasses

        available_metrics = {}
        for cls in find_descendants(BaseMetric):
            metric_name = cls.metric_name
            available_metrics[metric_name] = cls

        return available_metrics

    @abstractmethod
    def evaluate(self, data) -> Dict[str, float]:
        """
        执行评估（抽象方法，由 AnswerEvaluator 或 RetrievalEvaluator 具体实现）
        """
        pass

    def save_metric_score(self, result_dict: Dict[str, float]):
        """将最终的平均分保存为 txt 文本"""
        if not self.save_metric_flag:
            return

        file_name = "metric_score.txt"
        save_path = os.path.join(self.save_dir, file_name)
        with open(save_path, "w", encoding='utf-8') as f:
            for k, v in result_dict.items():
                f.write(f"{k}: {v}\n")

    def save_data(self, data):
        """将带有详细分数的中间数据保存为 json"""
        if not self.save_data_flag:
            return

        file_name = "intermediate_data.json"
        save_path = os.path.join(self.save_dir, file_name)

        # 优先调用 data 对象自带的 save 方法（比如 evaluation_data.py 里定义的）
        if hasattr(data, 'save'):
            data.save(save_path)
        else:
            try:
                serializable_data = self._convert_to_serializable(data)
                with open(save_path, "w", encoding='utf-8') as f:
                    json.dump(serializable_data, f, ensure_ascii=False, indent=2)
            except Exception as e:
                print(f"保存中间数据时出错: {e}")

    def _convert_to_serializable(self, data):
        """递归地将复杂的对象转换为可以被 json.dump 序列化的字典"""
        if isinstance(data, dict):
            return {k: self._convert_to_serializable(v) for k, v in data.items()}
        elif isinstance(data, list):
            return [self._convert_to_serializable(item) for item in data]
        elif hasattr(data, '__dict__'):
            return self._convert_to_serializable(data.__dict__)
        else:
            return data

    def format_results_table(self, results: Dict[str, float]) -> str:
        """将评估的字典结果一键转换为 Markdown 表格，方便输出和展示"""
        header = "| 指标 | 得分 |"
        separator = "| --- | --- |"

        rows = []
        for metric, score in results.items():
            if isinstance(score, float):
                score_str = f"{score:.4f}"
            else:
                score_str = str(score)
            rows.append(f"| {metric} | {score_str} |")

        table = "\n".join([header, separator] + rows)
        return table

    def log(self, message, *args, **kwargs):
        if self.debug:
            print(f"[{self.__class__.__name__}] {message}", *args, **kwargs)
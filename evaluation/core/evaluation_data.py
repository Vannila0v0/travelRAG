import json
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Any, Tuple

# 引入文本预处理工具（代码在下方提供）
from evaluation.preprocessing.text_cleaner import clean_thinking_process, clean_references
from evaluation.preprocessing.reference_extractor import extract_references_from_answer


class JsonSerializable:
    """可序列化为JSON的基类"""

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'JsonSerializable':
        """从字典创建实例"""
        return cls(**data)


@dataclass
class AnswerEvaluationSample(JsonSerializable):
    """答案评估样本类（为多智能体和缓存系统做了扩充）"""

    question: str
    golden_answer: str
    system_answer: str = ""
    scores: Dict[str, float] = field(default_factory=dict)
    agent_type: str = ""  # e.g., hybrid, multi_agent

    # --- [新增] 适配系统的专属字段 ---
    # 用于存放 agent_system 的中间日志 (如 plan_spec, report_spec)
    execution_trace: Dict[str, Any] = field(default_factory=dict)
    # 用于存放 cache_manager 的命中记录 (如 hit_level, time_saved)
    cache_info: Dict[str, Any] = field(default_factory=dict)

    # 兼容原版的图谱字段
    retrieved_entities: List[str] = field(default_factory=list)
    retrieved_relationships: List = field(default_factory=list)

    def update_system_answer(self, answer: str, agent_type: str = ""):
        """更新系统回答，自动清理引用数据和思考过程"""
        cleaned_answer = clean_thinking_process(answer)
        cleaned_answer = clean_references(cleaned_answer)

        self.system_answer = cleaned_answer
        if agent_type:
            self.agent_type = agent_type

    def update_evaluation_score(self, metric: str, score: float):
        """更新单个评估分数"""
        self.scores[metric] = score


@dataclass
class AnswerEvaluationData:
    """答案评估数据集管理类"""

    samples: List[AnswerEvaluationSample] = field(default_factory=list)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> AnswerEvaluationSample:
        return self.samples[idx]

    def append(self, sample: AnswerEvaluationSample):
        self.samples.append(sample)

    @property
    def questions(self) -> List[str]:
        return [sample.question for sample in self.samples]

    @property
    def golden_answers(self) -> List[str]:
        return [sample.golden_answer for sample in self.samples]

    @property
    def system_answers(self) -> List[str]:
        return [sample.system_answer for sample in self.samples]

    def save(self, path: str):
        """保存评估数据到 JSON"""
        with open(path, "w", encoding='utf-8') as f:
            json.dump([sample.to_dict() for sample in self.samples], f, ensure_ascii=False, indent=2)

    @classmethod
    def load(cls, path: str) -> 'AnswerEvaluationData':
        """从 JSON 加载评估数据"""
        with open(path, "r", encoding='utf-8') as f:
            samples_data = json.load(f)

        data = cls()
        for sample_data in samples_data:
            sample = AnswerEvaluationSample.from_dict(sample_data)
            data.append(sample)
        return data


@dataclass
class RetrievalEvaluationSample(JsonSerializable):
    """检索评估样本类"""

    question: str
    system_answer: str = ""
    retrieved_entities: List[str] = field(default_factory=list)
    retrieved_relationships: List[Tuple[str, str, str]] = field(default_factory=list)
    referenced_entities: List[str] = field(default_factory=list)
    referenced_relationships: List = field(default_factory=list)
    scores: Dict[str, float] = field(default_factory=dict)
    agent_type: str = ""
    retrieval_time: float = 0.0
    retrieval_logs: Dict[str, Any] = field(default_factory=dict)
    entity_details: List[Dict[str, str]] = field(default_factory=list)
    enhanced_relationships: List[Tuple[str, str, str]] = field(default_factory=list)

    # --- [新增] 适配你的系统的专属字段 ---
    execution_trace: Dict[str, Any] = field(default_factory=dict)
    cache_info: Dict[str, Any] = field(default_factory=dict)

    def update_system_answer(self, answer: str, agent_type: str = ""):
        """更新系统回答并提取回答中实际使用的引用 [Entity_xx]"""
        # 如果是具有复杂推理的Agent，先清理 <think> 过程
        if agent_type in ["deep", "multi_agent"]:
            answer = clean_thinking_process(answer)

        self.system_answer = answer
        if agent_type:
            self.agent_type = agent_type

        # 从纯文本中正则提取 [Entity_1], [Rel_2]
        refs = extract_references_from_answer(answer)
        self.referenced_entities = refs.get("entities", [])
        self.referenced_relationships = refs.get("relationships", [])

    def update_retrieval_data(self, entities: List[str], relationships: List[Tuple[str, str, str]]):
        self.retrieved_entities = entities
        self.retrieved_relationships = relationships

    def update_logs(self, logs: Dict[str, Any]):
        self.retrieval_logs = logs

    def update_evaluation_score(self, metric: str, score: float):
        self.scores[metric] = score

    def to_dict(self) -> Dict[str, Any]:
        """重写以处理复杂的图谱元组和LangChain消息对象"""
        result = asdict(self)

        # 将关系元组转换为列表，以便 JSON 序列化
        result["retrieved_relationships"] = [list(rel) for rel in self.retrieved_relationships]
        if hasattr(self, 'enhanced_relationships') and self.enhanced_relationships:
            result["enhanced_relationships"] = [list(rel) for rel in self.enhanced_relationships]

        # 处理可能嵌套的 LangChain Message 对象
        if "retrieval_logs" in result and isinstance(result["retrieval_logs"], dict):
            logs = result["retrieval_logs"]
            if "execution_log" in logs and isinstance(logs["execution_log"], list):
                for i, log in enumerate(logs["execution_log"]):
                    if "input" in log and hasattr(log["input"], "__class__") and "Message" in log[
                        "input"].__class__.__name__:
                        logs["execution_log"][i]["input"] = str(log["input"])
                    if "output" in log and hasattr(log["output"], "__class__") and "Message" in log[
                        "output"].__class__.__name__:
                        logs["execution_log"][i]["output"] = str(log["output"])

        return result


@dataclass
class RetrievalEvaluationData:
    """检索评估数据集管理类"""

    samples: List[RetrievalEvaluationSample] = field(default_factory=list)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> RetrievalEvaluationSample:
        return self.samples[idx]

    def append(self, sample: RetrievalEvaluationSample):
        self.samples.append(sample)

    def save(self, path: str):
        class CustomEncoder(json.JSONEncoder):
            def default(self, obj):
                try:
                    # 尝试序列化遗漏的 LangChain 对象
                    from langchain_core.messages import BaseMessage
                    if isinstance(obj, BaseMessage):
                        return str(obj)
                except ImportError:
                    pass
                return super().default(obj)

        with open(path, "w", encoding='utf-8') as f:
            samples_data = [sample.to_dict() for sample in self.samples]
            json.dump(samples_data, f, ensure_ascii=False, indent=2, cls=CustomEncoder)

    @classmethod
    def load(cls, path: str) -> 'RetrievalEvaluationData':
        with open(path, "r", encoding='utf-8') as f:
            samples_data = json.load(f)

        data = cls()
        for sample_data in samples_data:
            # 还原元组格式
            if "retrieved_relationships" in sample_data:
                sample_data["retrieved_relationships"] = [tuple(rel) for rel in sample_data["retrieved_relationships"]]
            if "enhanced_relationships" in sample_data:
                sample_data["enhanced_relationships"] = [tuple(rel) for rel in sample_data["enhanced_relationships"]]

            sample = RetrievalEvaluationSample(**sample_data)
            data.append(sample)
        return data
import re
import json
from typing import Dict, List, Tuple

# 引入基础指标类
from evaluation.core.base_metric import BaseMetric
# 引入引用提取工具
from evaluation.preprocessing.reference_extractor import extract_references_from_answer


class ResponseCoherence(BaseMetric):
    """回答连贯性评估指标 - 评估回答的结构化程度和逻辑清晰度"""

    metric_name = "response_coherence"

    def __init__(self, config):
        super().__init__(config)
        self.llm = config.get("llm", None)

    def calculate_metric(self, data) -> Tuple[Dict[str, float], List[float]]:
        self.log("\n======== ResponseCoherence (连贯性) 计算日志 ========")

        if not self.llm:
            self.log("错误: 未提供LLM模型，无法执行连贯性评估")
            return {"response_coherence": 0.0}, [0.0] * len(data.samples)

        coherence_scores = []

        for idx, sample in enumerate(data.samples):
            question = sample.question
            answer = sample.system_answer

            self.log(f"\n样本 {idx + 1}:")

            # 提取结构化特征，辅助 LLM 判断
            paragraphs = answer.split('\n\n')
            has_headers = bool(re.search(r'#{1,3}\s+\w+', answer))
            sentence_count = len(re.findall(r'[.!?。！？]\s*', answer))

            self.log(f"  结构特征 -> 段落数: {len(paragraphs)}, 包含标题: {has_headers}, 句子数: {sentence_count}")

            prompt = f"""
            评估以下回答的连贯性和结构，给出0到1的分数。
            评分标准:
            - 高分(0.8-1.0): 逻辑清晰，结构良好，使用标题和段落，思路连贯
            - 中分(0.4-0.7): 内容基本清晰，但可能存在一些逻辑跳跃
            - 低分(0.0-0.3): 结构混乱，缺乏逻辑性

            问题: {question}
            回答: {answer}

            只返回一个0到1之间的数字表示分数，不要有任何其他文字。
            """

            try:
                response = self.llm.invoke(prompt)
                score_text = response.content if hasattr(response, 'content') else str(response)

                score_match = re.search(r'(\d+(\.\d+)?)', score_text)
                if score_match:
                    coherence = float(score_match.group(1))
                    coherence = max(0.0, min(1.0, coherence))
                else:
                    coherence = 0.5
            except Exception as e:
                self.log(f"  LLM评估连贯性时出错: {e}")
                coherence = 0.5

            coherence_scores.append(coherence)
            self.log(f"  连贯性得分: {coherence:.4f}")

        avg_coherence = sum(coherence_scores) / len(coherence_scores) if coherence_scores else 0.0
        self.log(f"回答连贯性平均得分: {avg_coherence:.4f}")
        self.log("======== ResponseCoherence 计算结束 ========\n")

        return {"response_coherence": avg_coherence}, coherence_scores


class FactualConsistency(BaseMetric):
    """事实一致性评估指标 - 评估回答是否有幻觉，是否忠于检索到的上下文"""

    metric_name = "factual_consistency"

    def __init__(self, config):
        super().__init__(config)
        self.llm = config.get("llm", None)

    def calculate_metric(self, data) -> Tuple[Dict[str, float], List[float]]:
        self.log("\n======== FactualConsistency (事实一致性) 计算日志 ========")

        if not self.llm:
            self.log("错误: 未提供LLM模型，无法执行事实一致性评估")
            return {"factual_consistency": 0.0}, [0.0] * len(data.samples)

        consistency_scores = []

        for idx, sample in enumerate(data.samples):
            answer = sample.system_answer
            question = sample.question

            self.log(f"\n样本 {idx + 1}:")

            # 提取回答中的关键声明(Claims)
            key_facts = []
            for line in answer.split('\n'):
                if line.strip() and not line.startswith('#'):
                    stripped = line.strip('- *')
                    if len(stripped) > 10:
                        key_facts.append(stripped)
            facts_text = "\n".join([f"- {fact}" for fact in key_facts[:10]])

            # --- [为你新增的专属多智能体适配] ---
            # 如果你的 Agent System 留下了执行轨迹（比如 Executor 检索到的内容），把它喂给 LLM 做比对
            trace_context = ""
            if hasattr(sample, 'execution_trace') and sample.execution_trace:
                executor_log = sample.execution_trace.get('executor', '')
                if executor_log:
                    trace_context = f"\n智能体检索到的原始背景资料:\n{executor_log}\n"
                    self.log("  检测到多智能体执行轨迹，已注入提示词进行严谨比对。")
            # ----------------------------------

            prompt = f"""
            评估以下回答对问题的事实一致性，给出0到1的分数。
            评分标准:
            - 高分(0.8-1.0): 回答内容逻辑一致，信息准确，无矛盾内容。如果提供了背景资料，回答必须完全忠于资料，无捏造。
            - 中分(0.4-0.7): 回答大部分内容自洽，但有些模糊或可能不够精确。
            - 低分(0.0-0.3): 回答内容自相矛盾、明显错误或产生严重幻觉。

            问题: {question}
            {trace_context}
            回答的关键信息点:
            {facts_text}

            完整回答:
            {answer}

            只返回一个0到1之间的数字表示分数，不要有任何其他文字。
            """

            try:
                response = self.llm.invoke(prompt)
                score_text = response.content if hasattr(response, 'content') else str(response)

                score_match = re.search(r'(\d+(\.\d+)?)', score_text)
                if score_match:
                    consistency = float(score_match.group(1))
                    consistency = max(0.0, min(1.0, consistency))
                else:
                    consistency = 0.6
            except Exception as e:
                self.log(f"  LLM评估事实一致性时出错: {e}")
                consistency = 0.6

            consistency_scores.append(consistency)
            self.log(f"  事实一致性得分: {consistency:.4f}")

        avg_consistency = sum(consistency_scores) / len(consistency_scores) if consistency_scores else 0.0
        self.log(f"事实一致性平均得分: {avg_consistency:.4f}")
        self.log("======== FactualConsistency 计算结束 ========\n")

        return {"factual_consistency": avg_consistency}, consistency_scores


class ComprehensiveAnswerMetric(BaseMetric):
    """回答全面性评估指标 - 评估是否充分解答了 Prompt 中的所有疑问点"""

    metric_name = "answer_comprehensiveness"

    def __init__(self, config):
        super().__init__(config)
        self.llm = config.get("llm", None)

    def calculate_metric(self, data) -> Tuple[Dict[str, float], List[float]]:
        self.log("\n======== AnswerComprehensiveness (全面性) 计算日志 ========")

        if not self.llm:
            return {"answer_comprehensiveness": 0.0}, [0.0] * len(data.samples)

        comprehensiveness_scores = []

        for idx, sample in enumerate(data.samples):
            question = sample.question
            answer = sample.system_answer

            prompt = f"""
            评估以下回答解决问题的全面性，给出0到1的分数。
            评分标准:
            - 高分(0.8-1.0): 回答全面地解决了问题的所有方面，提供了丰富的信息和细节
            - 中分(0.4-0.7): 回答基本解决了问题，但可能遗漏了一些次要方面
            - 低分(0.0-0.3): 回答不完整，忽略了问题的主要方面

            问题: {question}
            回答: {answer}

            只返回一个0到1之间的数字表示分数，不要有任何其他文字。
            """

            try:
                response = self.llm.invoke(prompt)
                score_text = response.content if hasattr(response, 'content') else str(response)
                score_match = re.search(r'(\d+(\.\d+)?)', score_text)
                if score_match:
                    score = float(score_match.group(1))
                    score = max(0.0, min(1.0, score))
                else:
                    score = 0.5
            except Exception:
                score = 0.5

            comprehensiveness_scores.append(score)
            self.log(f"  样本 {idx + 1} 全面性得分: {score:.4f}")

        avg_score = sum(comprehensiveness_scores) / len(comprehensiveness_scores) if comprehensiveness_scores else 0.0
        self.log(f"回答全面性平均得分: {avg_score:.4f}")
        return {"answer_comprehensiveness": avg_score}, comprehensiveness_scores


class LLMGraphRagEvaluator(BaseMetric):
    """综合性 LLM 裁判 - 返回一个包含多维度的详细 JSON 字典"""

    metric_name = "llm_evaluation"

    def __init__(self, config):
        super().__init__(config)
        self.llm = config.get("llm", None)
        # 各维度的加权权重
        self.aspect_weights = {
            "comprehensiveness": 0.3,  # 全面性
            "relativeness": 0.25,  # 相关性
            "empowerment": 0.25,  # 增强理解能力
            "directness": 0.2  # 直接性
        }

    def calculate_metric(self, data) -> Tuple[Dict[str, float], List[Dict[str, float]]]:
        self.log("\n======== LLMGraphRagEvaluator (综合裁判) 计算日志 ========")

        if not self.llm:
            empty_scores = {f"llm_{aspect}": 0.0 for aspect in self.aspect_weights}
            empty_scores["llm_total"] = 0.0
            return empty_scores, [{} for _ in data.samples]

        all_scores = []
        summary_scores = {aspect: [] for aspect in self.aspect_weights}

        for idx, sample in enumerate(data.samples):
            question = sample.question
            # 清理可能的残余标签
            answer = re.sub(r'#{1,4}\s*引用数据[\s\S]*?(\{[\s\S]*?\})\s*$', '', sample.system_answer).rstrip()

            self.log(f"\n样本 {idx + 1}: 正在使用 LLM 进行多维度分析...")

            eval_prompt = f"""
            请评估以下回答相对于问题的质量，给出0到1之间的分数。

            1. 全面性(comprehensiveness): 回答涵盖了问题各个方面的程度
            2. 相关性(relativeness): 回答与问题的相关程度
            3. 增强理解能力(empowerment): 回答帮助读者理解并做出判断的程度
            4. 直接性(directness): 回答直接回应问题，不偏离主题的程度

            问题: {question}
            回答: {answer}

            请严格按照以下 JSON 格式返回结果（不要包含 Markdown 代码块标记如 ```json）：
            {{
                "comprehensiveness": 0.8,
                "relativeness": 0.9,
                "empowerment": 0.7,
                "directness": 0.85,
                "reasoning": "回答逻辑清晰..."
            }}
            """

            try:
                response = self.llm.invoke(eval_prompt)
                content = response.content if hasattr(response, 'content') else str(response)

                # 正则提取 JSON
                json_match = re.search(r'(\{[\s\S]*\})', content)
                if json_match:
                    data_json = json.loads(json_match.group(1))
                    sample_scores = {}
                    for aspect in self.aspect_weights:
                        val = data_json.get(aspect, 0.5)
                        score_value = min(1.0, max(0.0, float(val)))
                        sample_scores[aspect] = score_value
                        summary_scores[aspect].append(score_value)
                    all_scores.append(sample_scores)
                    self.log(f"  解析成功，评分理由: {data_json.get('reasoning', '无')}")
                else:
                    raise ValueError("JSON 提取失败")

            except Exception as e:
                self.log(f"  LLM评估出错: {e}，使用默认分数 0.5")
                default_scores = {aspect: 0.5 for aspect in self.aspect_weights}
                all_scores.append(default_scores)
                for aspect in self.aspect_weights:
                    summary_scores[aspect].append(0.5)

        # 统计所有样本的平均分和加权总分
        avg_scores = {}
        for aspect, scores in summary_scores.items():
            avg_scores[f"llm_{aspect}"] = sum(scores) / len(scores) if scores else 0.0

        weighted_sum = sum(avg_scores[f"llm_{aspect}"] * weight for aspect, weight in self.aspect_weights.items())
        avg_scores["llm_total"] = weighted_sum

        self.log(f"\n多维度加权总分: {weighted_sum:.4f}")
        self.log("======== LLMGraphRagEvaluator 计算结束 ========\n")

        return avg_scores, all_scores
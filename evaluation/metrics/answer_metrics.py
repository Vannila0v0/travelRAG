import re
from typing import Dict, List, Tuple

# 引入你之前创建的 BaseMetric
from evaluation.core.base_metric import BaseMetric
# 引入数据结构容器（你需要确保 evaluation_data.py 已迁移）
from evaluation.core.evaluation_data import AnswerEvaluationData
# 引入文本标准化工具
from evaluation.utils.text_utils import normalize_answer


class ExactMatch(BaseMetric):
    """精确匹配评估指标 (Exact Match)"""

    metric_name = "em"

    def __init__(self, config):
        super().__init__(config)
        self.llm = config.get("llm", None)

    def calculate_em(self, prediction: str, golden_answer: str) -> float:
        """计算单个预测的精确匹配得分"""
        if not prediction or not golden_answer:
            return 0.0

        normalized_prediction = normalize_answer(prediction)
        normalized_golden = normalize_answer(golden_answer)

        # 完全匹配
        if normalized_prediction == normalized_golden:
            return 1.0
        return 0.0

    def calculate_metric(self, data: AnswerEvaluationData) -> Tuple[Dict[str, float], List[float]]:
        """计算精确匹配指标 - 使用规则匹配和LLM回退混合评分"""
        self.log("======== ExactMatch 计算日志 ========")
        self.log(f"样本总数: {len(data.samples) if hasattr(data, 'samples') else 0}")

        golden_answers = data.golden_answers
        system_answers = data.system_answers

        metric_score_list = []

        for idx, (pred, golden) in enumerate(zip(system_answers, golden_answers)):
            # 预处理系统答案 - 移除Markdown标题和多余空行
            cleaned_pred = re.sub(r'^###.*?\n+', '', pred, flags=re.MULTILINE)
            cleaned_pred = re.sub(r'\n\s*\n', '\n', cleaned_pred)
            cleaned_pred = cleaned_pred.strip()

            # 标准化答案
            normalized_pred = normalize_answer(cleaned_pred)
            normalized_golden = normalize_answer(golden)

            self.log(f"\n样本 {idx + 1}:")
            self.log(f"  标准答案(前30字符): {golden[:30]}...")
            self.log(f"  清理后的系统答案(前30字符): {cleaned_pred[:30]}...")

            # 1. 完全匹配
            if normalized_pred == normalized_golden:
                score = 1.0
                self.log(f"  完全匹配 ✓")
            else:
                # 2. 规则匹配失败，尝试内容相似性评估
                similarity_score = self._calculate_content_similarity(cleaned_pred, golden)
                self.log(f"  基本内容相似度: {similarity_score:.4f}")

                # 如果内容相似度较高，给予一定折算分数
                if similarity_score >= 0.7:
                    score = 0.7 + (similarity_score - 0.7) * 3 / 3
                    self.log(f"  内容高度相似，给予分数: {score:.4f}")
                # 3. 如果内容相似度一般，回退到 LLM 评分 (LLM Fallback)
                elif self.llm:
                    self.log(f"  内容相似度一般，回退到LLM评分")
                    prompt = f"""
                    请比较下面两个答案，评估它们在内容上的等价性，给出0到1之间的分数。
                    0表示完全不同，1表示内容上完全等价。
                    请只考虑实质内容，忽略格式、表达方式和顺序的差异。

                    标准答案:
                    {golden}

                    系统答案:
                    {cleaned_pred}

                    只返回一个0到1之间的数字表示分数，不要有任何其他文字。
                    """
                    # 调用 BaseMetric 中的回退打分机制
                    score = self.get_llm_fallback_score(prompt, default_score=similarity_score)
                    self.log(f"  LLM评估的匹配度分数: {score:.4f}")
                else:
                    # 没有LLM，只能使用内容相似度作为分数
                    score = similarity_score
                    self.log(f"  使用内容相似度作为分数: {score:.4f}")

            metric_score_list.append(score)

        em_score = sum(metric_score_list) / len(metric_score_list) if metric_score_list else 0.0
        self.log(f"\n匹配样本数: {sum(1 for s in metric_score_list if s > 0.8)}")
        self.log(f"精确匹配平均得分: {em_score:.4f}")
        self.log("======== ExactMatch 计算结束 ========\n")

        return {"em": em_score}, metric_score_list

    def _calculate_content_similarity(self, pred: str, golden: str) -> float:
        """计算两个文本的基础内容相似度 (Jaccard + 词覆盖率)"""
        pred_norm = normalize_answer(pred).split()
        golden_norm = normalize_answer(golden).split()

        if not pred_norm or not golden_norm:
            return 0.0

        common_words = set(pred_norm) & set(golden_norm)
        union_words = set(pred_norm) | set(golden_norm)

        jaccard = len(common_words) / len(union_words) if union_words else 0.0
        pred_coverage = len(common_words) / len(set(pred_norm)) if pred_norm else 0
        golden_coverage = len(common_words) / len(set(golden_norm)) if golden_norm else 0

        # 综合得分 - Jaccard占40%，两个覆盖率各占30%
        similarity = 0.4 * jaccard + 0.3 * pred_coverage + 0.3 * golden_coverage
        return similarity


class F1Score(BaseMetric):
    """F1分数评估指标 (F1 Score)"""

    metric_name = "f1"

    def __init__(self, config):
        super().__init__(config)
        self.llm = config.get("llm", None)

    def calculate_metric(self, data: AnswerEvaluationData) -> Tuple[Dict[str, float], List[float]]:
        """计算F1分数 - 使用规则匹配和LLM回退混合评分"""
        self.log("\n======== F1Score 计算日志 ========")
        self.log(f"样本总数: {len(data.samples) if hasattr(data, 'samples') else 0}")

        golden_answers = data.golden_answers
        system_answers = data.system_answers

        f1_scores = []

        for idx, (pred, golden) in enumerate(zip(system_answers, golden_answers)):
            cleaned_pred = re.sub(r'^###.*?\n+', '', pred, flags=re.MULTILINE)
            cleaned_pred = re.sub(r'\n\s*\n', '\n', cleaned_pred).strip()

            pred_text = normalize_answer(cleaned_pred)
            golden_text = normalize_answer(golden)

            self.log(f"\n样本 {idx + 1}:")

            try:
                # 1. 尝试使用 jieba 进行传统中文分词与 F1 计算
                import jieba
                pred_tokens = list(jieba.cut(pred_text))
                golden_tokens = list(jieba.cut(golden_text))

                stopwords = {'的', '了', '和', '在', '是', '为', '以', '与', '或', '且'}
                pred_tokens = [token for token in pred_tokens if len(token) > 1 and token not in stopwords]
                golden_tokens = [token for token in golden_tokens if len(token) > 1 and token not in stopwords]

                if not pred_tokens or not golden_tokens:
                    rule_f1 = 1.0 if not pred_tokens and not golden_tokens else 0.0
                else:
                    common_tokens = set(pred_tokens) & set(golden_tokens)
                    precision = len(common_tokens) / len(pred_tokens) if pred_tokens else 0
                    recall = len(common_tokens) / len(golden_tokens) if golden_tokens else 0

                    if precision + recall > 0:
                        rule_f1 = 2 * precision * recall / (precision + recall)
                    else:
                        rule_f1 = 0.0

                self.log(f"  规则F1分数: {rule_f1:.4f}")
            except Exception as e:
                self.log(f"  规则F1计算出错: {e}")
                rule_f1 = 0.0

            # 2. 无论规则 F1 表现如何，如果有 LLM，都并行让 LLM 进行语义打分
            if self.llm:
                prompt = f"""
                请比较下面两个答案的内容相似度，评估它们包含的信息重叠程度，并给出0到1之间的分数。
                0表示完全不同信息，1表示信息完全重叠。
                请考虑实质内容的相似性，而不仅是表面文字的匹配。在评估时，请特别关注关键信息点是否一致。

                标准答案:
                {golden}

                系统答案:
                {cleaned_pred}

                只返回一个0到1之间的数字表示分数，不要有任何其他文字。
                """

                llm_f1 = self.get_llm_fallback_score(prompt, default_score=0.5)
                self.log(f"  LLM评估的F1分数: {llm_f1:.4f}")

                # 3. 择优录取：取规则 F1 和 LLM 语义 F1 中的最高值
                f1 = max(llm_f1, rule_f1)
                self.log(f"  最终采用分数: {f1:.4f}")
            else:
                f1 = rule_f1

            f1_scores.append(f1)

        avg_f1 = sum(f1_scores) / len(f1_scores) if f1_scores else 0.0
        self.log(f"\nF1平均得分: {avg_f1:.4f}")
        self.log("======== F1Score 计算结束 ========\n")

        return {"f1": avg_f1}, f1_scores
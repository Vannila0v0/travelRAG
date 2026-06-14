import re
from typing import Dict, List, Tuple

# 引入你的基类和数据容器
from evaluation.core.base_metric import BaseMetric
from evaluation.core.evaluation_data import RetrievalEvaluationData
# 引入预处理工具
from evaluation.preprocessing.reference_extractor import extract_references_from_answer
from evaluation.preprocessing.text_cleaner import clean_references, clean_thinking_process


class RetrievalPrecision(BaseMetric):
    """检索精确率评估指标 - 评估回答中引用的实体，有多少是真的被检索出来的（防捏造/幻觉）"""

    metric_name = "retrieval_precision"

    def __init__(self, config):
        super().__init__(config)
        self.neo4j_client = config.get('neo4j_client', None)
        self.llm = config.get("llm", None)

    def calculate_metric(self, data: RetrievalEvaluationData) -> Tuple[Dict[str, float], List[float]]:
        self.log("\n======== RetrievalPrecision (检索精确率) 计算日志 ========")

        retrieved_entities = data.retrieved_entities
        referenced_entities = data.referenced_entities

        precision_scores = []
        for idx, (retr_entities, ref_entities) in enumerate(zip(retrieved_entities, referenced_entities)):
            # 基础防御，如果没有引用或没有检索到，给基础分或交由 LLM 判断
            if not retr_entities or not ref_entities:
                base_score = 0.3
                if self.llm:
                    llm_score = self._get_llm_precision_score(data.samples[idx], retr_entities, ref_entities)
                    precision_scores.append(max(base_score, llm_score))
                else:
                    precision_scores.append(base_score)
                continue

            # 规则匹配评分
            matched, rule_score = self._calculate_rule_precision(retr_entities, ref_entities)

            # 规则评分不佳时，触发 LLM 回退裁判
            if rule_score <= 0.5 and self.llm:
                llm_score = self._get_llm_precision_score(data.samples[idx], retr_entities, ref_entities)
                precision_scores.append(max(rule_score, llm_score))
            else:
                precision_scores.append(rule_score)

        avg_precision = sum(precision_scores) / len(precision_scores) if precision_scores else 0.3
        self.log(f"检索精确率平均得分: {avg_precision:.4f}")
        return {"retrieval_precision": avg_precision}, precision_scores

    def _calculate_rule_precision(self, retr_entities, ref_entities):
        """计算规则匹配精确率"""
        retr_entities_str = [str(e).lower() for e in retr_entities]
        ref_entities_str = [str(e).lower() for e in ref_entities]

        direct_matches = sum(1 for ref_id in ref_entities_str if any(ref_id in retr for retr in retr_entities_str))
        num_matches = 0
        for ref_id in ref_entities_str:
            ref_num = re.search(r'\d+', ref_id)
            if ref_num and any(ref_num.group() in retr for retr in retr_entities_str):
                num_matches += 1

        matched = max(direct_matches, num_matches)
        if matched > 0:
            return matched, max(0.3, 0.3 + 0.7 * (matched / len(ref_entities_str)))
        return 0, 0.3

    def _get_llm_precision_score(self, sample, retr_entities, ref_entities) -> float:
        retr_str = ", ".join([str(e) for e in retr_entities[:10]]) if retr_entities else "无"
        ref_str = ", ".join([str(e) for e in ref_entities[:10]]) if ref_entities else "无"

        prompt = f"""
        请评估检索到的实体与用户引用实体的匹配程度，给出0到1的分数。
        检索到的实体: [{retr_str}]
        用户引用的实体: [{ref_str}]
        回答(部分): {sample.system_answer[:150]}...
        只返回一个0到1之间的数字。
        """
        return self.get_llm_fallback_score(prompt, default_score=0.4)


class RetrievalUtilization(BaseMetric):
    """检索利用率评估指标 - 评估系统检索出来的一大堆实体，有多少被真正写进了答案里"""

    metric_name = "retrieval_utilization"

    def __init__(self, config):
        super().__init__(config)
        self.llm = config.get("llm", None)

    def calculate_metric(self, data: RetrievalEvaluationData) -> Tuple[Dict[str, float], List[float]]:
        self.log("\n======== RetrievalUtilization (检索利用率) 计算日志 ========")

        retrieved_entities = data.retrieved_entities
        referenced_entities = data.referenced_entities

        utilization_scores = []
        for idx, (retr_entities, ref_entities) in enumerate(zip(retrieved_entities, referenced_entities)):
            if not ref_entities or not retr_entities:
                base_score = 0.3
                if self.llm:
                    llm_score = self._get_llm_utilization_score(data.samples[idx], retr_entities, ref_entities)
                    utilization_scores.append(max(base_score, llm_score))
                else:
                    utilization_scores.append(base_score)
                continue

            # 规则匹配
            matches_found, rule_score = self._calculate_rule_utilization(retr_entities, ref_entities)

            if rule_score <= 0.5 and self.llm:
                llm_score = self._get_llm_utilization_score(data.samples[idx], retr_entities, ref_entities)
                utilization_scores.append(max(rule_score, llm_score))
            else:
                utilization_scores.append(rule_score)

        avg_utilization = sum(utilization_scores) / len(utilization_scores) if utilization_scores else 0.3
        self.log(f"检索利用率平均得分: {avg_utilization:.4f}")
        return {"retrieval_utilization": avg_utilization}, utilization_scores

    def _calculate_rule_utilization(self, retr_entities, ref_entities):
        retr_norm = [str(e).lower() for e in retr_entities]
        ref_norm = [str(e).lower() for e in ref_entities]

        direct_matches = sum(1 for ref_id in ref_norm if any(ref_id in retr for retr in retr_norm))
        matched = direct_matches

        if matched > 0:
            return matched, max(0.3, 0.3 + 0.7 * (matched / len(ref_norm)))
        return 0, 0.3

    def _get_llm_utilization_score(self, sample, retr_entities, ref_entities) -> float:
        prompt = f"""
        请评估系统在回答用户问题时对检索实体的利用程度，给出0到1的分数。
        检索到的实体: [{", ".join([str(e) for e in retr_entities[:10]])}]
        引用的实体: [{", ".join([str(e) for e in ref_entities[:10]])}]
        系统回答: {sample.system_answer[:200]}...
        只返回0到1之间的数字。
        """
        return self.get_llm_fallback_score(prompt, default_score=0.4)


class RetrievalLatency(BaseMetric):
    """检索延迟评估指标"""

    metric_name = "retrieval_latency"

    def __init__(self, config):
        super().__init__(config)

    def calculate_metric(self, data: RetrievalEvaluationData) -> Tuple[Dict[str, float], List[float]]:
        latency_scores = [sample.retrieval_time for sample in data.samples]
        avg_latency = sum(latency_scores) / len(latency_scores) if latency_scores else 0.0
        self.log(f"检索平均延迟: {avg_latency:.4f}秒")
        return {"retrieval_latency": avg_latency}, latency_scores


class ChunkUtilization(BaseMetric):
    """文本块利用率评估指标 (依赖图数据库客户端)"""

    metric_name = "chunk_utilization"

    def __init__(self, config):
        super().__init__(config)
        self.neo4j_client = config.get('neo4j_client', None)
        self.llm = config.get("llm", None)

    def calculate_metric(self, data: RetrievalEvaluationData) -> Tuple[Dict[str, float], List[float]]:
        self.log("\n======== ChunkUtilization (文本块利用率) 计算日志 ========")
        chunk_scores = []

        for sample in data.samples:
            refs = extract_references_from_answer(sample.system_answer)
            chunk_ids = refs.get("chunks", [])

            if not chunk_ids or not self.neo4j_client:
                score = 0.4
                if self.llm:
                    score = max(score, self._llm_fallback_for_chunk(sample, chunk_ids))
                chunk_scores.append(score)
                continue

            try:
                answer_text = clean_thinking_process(clean_references(sample.system_answer))
                total_matches = 0
                valid_chunks = 0

                for chunk_id in chunk_ids:
                    # 注意：如果你的图引擎 schema.py 里的标签不是 __Chunk__，请在这里修改！
                    query = "MATCH (n:__Chunk__) WHERE n.id = $id RETURN n.text AS text"
                    result = self.neo4j_client.execute_query(query, {"id": chunk_id})

                    if result.records:
                        chunk_text = result.records[0].get("text", "")
                        key_phrases = [p for p in re.findall(r'\b[\w\u4e00-\u9fa5]{4,}\b', chunk_text) if len(p) > 3]
                        if key_phrases:
                            matched = sum(1 for phrase in set(key_phrases) if phrase.lower() in answer_text.lower())
                            total_matches += matched / len(set(key_phrases))
                            valid_chunks += 1

                utilization = total_matches / valid_chunks if valid_chunks > 0 else 0.3
                chunk_scores.append(utilization)
            except Exception as e:
                self.log(f"读取Neo4j计算Chunk利用率时出错: {e}")
                chunk_scores.append(0.4)

        avg_chunk = sum(chunk_scores) / len(chunk_scores) if chunk_scores else 0.0
        return {"chunk_utilization": avg_chunk}, chunk_scores

    def _llm_fallback_for_chunk(self, sample, chunk_ids) -> float:
        prompt = f"请评估AI回答对 {len(chunk_ids)} 个检索文本块的利用程度...\n回答: {sample.system_answer[:200]}\n只返回0到1数字。"
        return self.get_llm_fallback_score(prompt, default_score=0.4)


# =====================================================================
# 🚀 独家定制：为你系统里的 Cache Manager 量身打造的专属指标
# =====================================================================
class CachePerformanceMetric(BaseMetric):
    """缓存性能评估指标 - 专门评估你系统中的 Cache Manager 表现"""

    metric_name = "cache_performance"

    def __init__(self, config):
        super().__init__(config)

    def calculate_metric(self, data: RetrievalEvaluationData) -> Tuple[Dict[str, float], List[Dict[str, float]]]:
        self.log("\n======== CachePerformanceMetric (缓存系统性能) 计算日志 ========")

        hit_count = 0
        total_time_saved = 0.0
        cache_details = []

        for idx, sample in enumerate(data.samples):
            # 获取我们在 evaluation_data.py 中为你新增的 cache_info 字段
            cache_info = getattr(sample, 'cache_info', {})

            is_hit = cache_info.get('hit', False)
            hit_level = cache_info.get('hit_level', 'none')  # e.g., memory, disk, vector
            time_saved = cache_info.get('time_saved_ms', 0) / 1000.0  # 转换为秒

            if is_hit:
                hit_count += 1
                total_time_saved += time_saved
                self.log(f"  样本 {idx + 1}: 命中缓存 [{hit_level}]，节省了 {time_saved:.3f} 秒")
            else:
                self.log(f"  样本 {idx + 1}: 未命中缓存 (Cache Miss)")

            cache_details.append({
                "hit": 1.0 if is_hit else 0.0,
                "time_saved_sec": time_saved
            })

        total_samples = len(data.samples)
        hit_rate = hit_count / total_samples if total_samples > 0 else 0.0
        avg_time_saved = total_time_saved / hit_count if hit_count > 0 else 0.0

        self.log(f"\n缓存总命中率: {hit_rate * 100:.1f}% ({hit_count}/{total_samples})")
        self.log(f"命中时平均节省时间: {avg_time_saved:.3f} 秒/次")
        self.log("======== CachePerformanceMetric 计算结束 ========\n")

        # 返回综合字典
        return {
            "cache_hit_rate": hit_rate,
            "cache_avg_time_saved_sec": avg_time_saved
        }, cache_details
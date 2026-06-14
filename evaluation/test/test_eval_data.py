import json
# 导入我们刚刚写好的数据结构
from evaluation.core.evaluation_data import AnswerEvaluationSample


def test_answer_sample():
    print("=== 开始测试 AnswerEvaluationSample ===\n")

    # 1. 模拟你的多智能体和缓存系统产生的数据
    mock_execution_trace = {
        "planner": "将问题拆解为: 1. 查询概念, 2. 总结特点",
        "executor": "执行了 2 次图谱检索，共耗时 1.2s",
        "reporter": "生成了最终的对比报告"
    }

    mock_cache_info = {
        "hit": True,
        "hit_level": "vector_cache",
        "time_saved_ms": 850
    }

    # 2. 实例化样本对象 (把你的中间数据塞进去)
    sample = AnswerEvaluationSample(
        question="什么是知识图谱？",
        golden_answer="知识图谱是一种用图模型来描述知识和建模世界万物之间关联关系的技术。",
        execution_trace=mock_execution_trace,
        cache_info=mock_cache_info
    )

    # 3. 模拟大模型生成的带有“碎碎念”和“引用标签”的原始回答
    raw_system_answer = """<think>
这个问题比较基础。我需要先从图数据库里找到知识图谱的定义。
好的，找到了节点 [Entity_001] 和 [Entity_005]。
接下来组织语言回答用户。
</think>
知识图谱是一种用图模型来描述知识和建模世界万物之间关联关系的技术 [Entity_001]。它通常由节点和边组成 [1][2]。"""

    print("【原始大模型回答】:")
    print(raw_system_answer)
    print("-" * 50)

    # 4. 调用更新方法 (这一步会自动触发文本清洗)
    sample.update_system_answer(answer=raw_system_answer, agent_type="multi_agent")

    # 5. 模拟打分 (假装咱们的 AnswerEvaluator 给它打了分)
    sample.update_evaluation_score("em", 1.0)
    sample.update_evaluation_score("f1", 0.95)

    # 6. 打印清洗后的结果和最终导出的字典
    print("\n【清洗后的干净回答 (用于算分)】:")
    print(sample.system_answer)
    print("-" * 50)

    print("\n【最终导出为 JSON 的格式】:")
    print(json.dumps(sample.to_dict(), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    test_answer_sample()
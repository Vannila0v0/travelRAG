# test_cache.py
import logging
from agent_system.cache_orchestrator import MultiAgentOrchestrator  # 注意类名变化
import time
# 确保日志显示
logging.basicConfig(level=logging.INFO)

# 初始化 (参数都是可选的)
agent = MultiAgentOrchestrator(enable_cache=True, cache_difficulty='simple')

print("\n=== 第一次运行 (First Run) ===")
start1_time = time.time()
state1 = agent.run("我想在三天内游玩象鼻山、两江四湖、漓江  ，该怎么安排？")
print(f"Result 1: {str(state1.final_report)}")
end1_time = time.time()
start2_time = time.time()
print("\n=== 第二次运行 (Second Run - Should hit cache) ===")
state2 = agent.run("我想在三天内游玩象鼻山、两江四湖、漓江，该怎么安排？")
print(f"Result 2: {str(state2.final_report)}")
end2_time = time.time()
print("first run consume = ", end2_time - start1_time, "s econd run consume = ", end2_time - start2_time)
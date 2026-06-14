import sys
import os

# 添加项目根目录到路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from agent_system.orchestrator import MultiAgentOrchestrator


def main():
    print("=== GraphRAG Agent (Plan-Execute-Report Architecture) ===")

    # 初始化总控
    orchestrator = MultiAgentOrchestrator()

    while True:
        query = input("\n🔎 请输入复杂查询 (输入 exit 退出): ").strip()
        if query.lower() == "exit":
            break
        if not query:
            continue

        print("-" * 60)

        # 运行编排器
        final_state = orchestrator.run(query)

        print("-" * 60)
        if final_state.final_report:
            print("\n📄 [最终报告]:\n")
            print(final_state.final_report)
        else:
            print("\n❌ 未能生成报告，请检查日志。")


if __name__ == "__main__":
    main()
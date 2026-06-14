import asyncio
import time

# 引入你写好的 HybridAgent
# 注意：请根据你的实际目录结构调整 import 路径
from graphrag_agent.agents.hybrid_agent import HybridAgent


# ==========================================
# 1. 模拟环境：Mock Neo4j 和 Vector Database
# ==========================================
class MockNeo4jHandler:
    def get_local_context(self, entities):
        """模拟 Local Search: 图谱局部查询"""
        return f"【图谱精确匹配】找到实体 {entities}：两江四湖夜游门票的官方挂牌价为 210 元，开放时间为每天 18:00 - 22:30。"

    @property
    def driver(self):
        """模拟 Global Search: 图谱全局社区查询 (Neo4j Session)"""

        class MockSession:
            def __enter__(self): return self

            def __exit__(self, exc_type, exc_val, exc_tb): pass

            def run(self, cypher_query):
                # 模拟图谱返回了几个长篇社区摘要
                return [
                    {"lvl": 1, "title": "夜游经济发展趋势",
                     "summary": "近年来，桂林夜间旅游经济持续升温，其中两江四湖作为核心夜游项目，游客量年均增长30%。"},
                    {"lvl": 0, "title": "购票渠道与优惠",
                     "summary": "线上OTA平台预订通常会有小幅度折扣，且节假日往往一票难求，建议游客提前一周预约。"}
                ]

        class MockDriver:
            def session(self): return MockSession()

        return MockDriver()


def mock_vector_search(query):
    """模拟 Local Search: 向量库检索"""

    class MockDoc:
        def __init__(self, content):
            self.page_content = content

    return [
        MockDoc("【文档库片段1】：两江四湖景区对1.2米以下儿童免票，1.2-1.4米儿童可享受半价。"),
        MockDoc("【文档库片段2】：游船全程大约需要 90 分钟，登船码头通常位于日月双塔附近。")
    ]


# ==========================================
# 2. 主测试流程
# ==========================================
async def main():
    print("====== 🚀 正在启动 HybridAgent 引擎 ======")

    # 初始化 Mock 数据库句柄
    neo4j_mock = MockNeo4jHandler()
    vector_mock = mock_vector_search

    # 初始化 Agent（开启 memory_only 方便测试缓存）
    # 注意：这里会自动触发 llm_factory 加载你的大模型
    agent = HybridAgent(
        neo4j_handler=neo4j_mock,
        vector_search_func=vector_mock
    )

    # 我们设计一个极其刁钻的问题：既问宏观趋势，又问微观细节
    query = "请结合当前夜游经济的发展趋势，详细告诉我两江四湖的夜游门票价格、儿童票政策以及开放时间？"

    print(f"\n🙋‍♂️ 用户提问: {query}\n")
    print("🤖 Agent 第一次思考与生成 (触发完整工作流)：")
    print("-" * 60)

    start_time = time.time()

    # 调用底层的 LangGraph 异步流！
    async for chunk in agent.ask_stream(query, thread_id="test_session_001"):
        # 这里的 chunk 会包含我们之前在 BaseAgent 里写的状态透出：
        # "🤔 **Agent正在分析问题**..."
        # "🔍 **正在检索本地知识图谱与向量库**..."
        # 以及最后大模型的流式打字机输出
        print(chunk, end="", flush=True)

    first_duration = time.time() - start_time
    print(f"\n\n⏱️ 第一次请求耗时: {first_duration:.2f} 秒")
    print("-" * 60)

    # ==========================================
    # 3. 测试双重缓存机制
    # ==========================================
    print("\n⚡️ 缓存测试 (第二次提问完全相同的问题)：")
    print("-" * 60)

    start_time_2 = time.time()

    async for chunk in agent.ask_stream(query, thread_id="test_session_001"):
        print(chunk, end="", flush=True)

    second_duration = time.time() - start_time_2
    print(f"\n\n⏱️ 第二次请求耗时 (命中缓存): {second_duration:.2f} 秒")
    print("-" * 60)


if __name__ == "__main__":
    # Windows 平台如果遇到 asyncio 报错，可以取消下面这行的注释
    # import asyncio; asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

    asyncio.run(main())
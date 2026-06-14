import asyncio
from agent_system.integration.llm_factory import get_llm_model
from graphrag_agent.Tools.LocalsearchTool import LocalSearchTool
from graphrag_agent.Tools.GlobalsearchTool import GlobalSearchTool


# ==========================================
# 模拟环境：Mock Neo4j 和 Vector Database
# ==========================================
class MockNeo4jHandler:
    def get_local_context(self, entities):
        """模拟图谱局部查询"""
        return f"这是来自图谱的 Mock 数据：实体 {entities} 的相关信息是开放时间 9:00-18:00，属于热门景区。"

    @property
    def driver(self):
        """模拟图谱全局社区查询 (Neo4j Session)"""

        class MockSession:
            def __enter__(self): return self

            def __exit__(self, exc_type, exc_val, exc_tb): pass

            def run(self, cypher_query):
                # 模拟图谱返回了几个长篇社区摘要
                return [
                    {"lvl": 2, "title": "大交通与旅游趋势",
                     "summary": "近年来，随着高铁和航空的便利，周末短途游和夜游项目成为核心增长点。尤其是夜游船票销量大幅上升。"},
                    {"lvl": 1, "title": "两江四湖夜游",
                     "summary": "两江四湖是城市名片，夜间灯光秀吸引了大量游客。成人票通常在200元左右波动。"},
                    {"lvl": 0, "title": "票务细节与优惠",
                     "summary": "部分渠道会有儿童半价票和老年人优惠政策，但节假日通常无折扣。"}
                ]

        class MockDriver:
            def session(self): return MockSession()

        return MockDriver()


def mock_vector_search(query):
    """模拟向量库检索"""

    class MockDoc:
        def __init__(self, content):
            self.page_content = content

    return [MockDoc("向量库片段1：两江四湖成人票挂牌价约为210元。"),
            MockDoc("向量库片段2：建议提前一天在网上预订，可能会有10-20元的优惠。")]


# ==========================================
# 主测试流程
# ==========================================
async def main():
    print("====== 1. 初始化模型与资源 ======")
    llm = get_llm_model()

    # ---------------------------------------------------------
    # 💡 如果你想测试真实数据，请把下面两行换成你真实的实例化代码：
    # from your_project.database import my_neo4j_handler
    # from your_project.vector import my_vector_search
    # neo4j_handler = my_neo4j_handler
    # vector_func = my_vector_search
    # ---------------------------------------------------------
    neo4j_handler = MockNeo4jHandler()
    vector_func = mock_vector_search

    # 初始化工具
    local_tool = LocalSearchTool(llm=llm, neo4j_handler=neo4j_handler, vector_search_func=vector_func)
    global_tool = GlobalSearchTool(llm=llm, neo4j_handler=neo4j_handler)

    print("\n====== 2. 测试 LocalSearchTool (局部精确搜索) ======")
    local_query = "夜游两江四湖的成人票多少钱？"
    try:
        # LocalTool 使用同步搜索即可
        local_result = local_tool.search(local_query)
        print("\n✅ [Local 最终返回给 Agent 的 Context]:")
        print(local_result)
    except Exception as e:
        print(f"❌ LocalSearchTool 测试失败: {e}")

    print("\n" + "=" * 50)

    print("\n====== 3. 测试 GlobalSearchTool (全局 Map-Reduce 搜索) ======")
    global_query = "请分析一下旅游景点的整体发展趋势和门票相关政策？"
    try:
        # GlobalTool 涉及大量的 LLM 并发调用，必须使用 await 异步执行
        global_result = await global_tool.asearch(global_query)
        print("\n✅ [Global 最终返回给 Agent 的 Context]:")
        print(global_result)
    except Exception as e:
        print(f"❌ GlobalSearchTool 测试失败: {e}")


if __name__ == "__main__":
    # Windows 下如果遇到 asyncio 报错，可以加上这行
    # import asyncio; asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    asyncio.run(main())
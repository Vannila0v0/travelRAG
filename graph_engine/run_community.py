import sys
import os
import networkx as nx
from collections import defaultdict

# 路径 Hack
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.neo4j_handler import Neo4jHandler
from graph_engine.community.detector import LeidenDetector

# 配置
NEO4J_URI = "bolt://localhost:7687"
NEO4J_USER = "neo4j"
NEO4J_PASSWORD = "password123"


def main():
    print("=== 开始执行 GraphRAG 社区发现 (Modular Version) ===")

    # 1. 初始资源
    handler = Neo4jHandler(NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD)
    detector = LeidenDetector(max_cluster_size=10)  # 实例化我们刚写的检测器

    # 2. 从 Neo4j 加载图 (NetworkX)
    print("1. Loading Graph from Neo4j...")
    G = nx.Graph()
    cypher = "MATCH (s:Entity)-[r]->(t:Entity) RETURN s.name as src, t.name as dst"

    with handler.driver.session() as session:
        res = session.run(cypher)
        for record in res:
            G.add_edge(record['src'], record['dst'])

    print(f"   -> Graph Loaded: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")

    if G.number_of_nodes() == 0:
        print("❌ 图为空，请先运行 builder.py")
        return

    # 3. 运行检测算法 (Detect)
    print("2. Running Community Detection...")
    partition = detector.detect(G)

    # 统计
    num_communities = len(set(partition.values()))
    print(f"   -> Found {num_communities} communities.")

    # 4. 写回 Neo4j (Write Back)
    print("3. Writing back to Neo4j...")

    # 准备批处理数据
    community_map = defaultdict(list)
    for node, cid in partition.items():
        community_map[cid].append(node)

    with handler.driver.session() as session:
        # 清理旧数据
        session.run("MATCH (c:Community) DETACH DELETE c")
        session.run("MATCH (e:Entity) REMOVE e.community_id")

        # 批量写入
        cypher_write = """
        UNWIND $batch AS row
        MERGE (c:Community {id: row.cid})
        SET c.level = 0
        WITH c, row
        MATCH (e:Entity {name: row.node})
        SET e.community_id = row.cid
        MERGE (c)-[:HAS_MEMBER]->(e)
        """

        batch = []
        for cid, nodes in community_map.items():
            for node in nodes:
                batch.append({"cid": int(cid), "node": node})
                if len(batch) >= 1000:
                    session.run(cypher_write, batch=batch)
                    batch = []
        if batch:
            session.run(cypher_write, batch=batch)

    print("✅ 社区发现流程结束！")
    handler.close()


if __name__ == "__main__":
    main()
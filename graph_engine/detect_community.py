import sys
import os
import networkx as nx
from collections import defaultdict

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from core.neo4j_handler import Neo4jHandler

# 配置
NEO4J_URI = "bolt://localhost:7687"
NEO4J_USER = "neo4j"
NEO4J_PASSWORD = "password123"


def run_hierarchical_community_detection():
    print("=== Step 4: Hierarchical Community Detection (层级社区发现) ===")

    handler = Neo4jHandler(NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD)

    # 1. 加载图
    print("1. Loading Graph...")
    G = nx.Graph()
    cypher_load = "MATCH (s:Entity)-[r]->(t:Entity) RETURN s.name AS src, t.name AS dst"
    with handler.driver.session() as session:
        result = session.run(cypher_load)
        for record in result:
            G.add_edge(record['src'], record['dst'])

    print(f"   -> Nodes: {G.number_of_nodes()}, Edges: {G.number_of_edges()}")

    if G.number_of_nodes() == 0:
        print("❌ 图为空，请先运行 builder.py")
        return

    # 2. 运行层级 Leiden
    print("2. Running Hierarchical Leiden...")
    try:
        from graspologic.partition import hierarchical_leiden

        # 确保节点顺序固定，因为 hierarchy 是个列表，通过下标索引
        node_list = list(G.nodes())

        # 运行算法
        hierarchy = hierarchical_leiden(G, max_cluster_size=10)

    except ImportError:
        print("❌ 必须安装 graspologic: pip install graspologic")
        return
    except Exception as e:
        print(f"❌ 算法运行失败: {e}")
        return

    # 3. 解析层级结构
    print("3. Parsing Hierarchy...")

    # 存储结构: { (level, community_id): [node1, node2...] }
    community_mapping = defaultdict(list)

    for i, cluster_obj in enumerate(hierarchy):
        node_name = node_list[i]

        # 遍历该节点所属的每一层社区
        # graspologic 的结构链: cluster -> parent_cluster -> parent_cluster
        current = cluster_obj
        while current:
            # === 核心修正区 ===
            # 不同版本的 graspologic 属性名可能不同，这里使用最标准的属性名
            # 1. 获取层级 (level)
            lvl = current.level

            # 2. 获取社区ID (cluster)
            # 报错提示 'cluster_id' 不存在，建议用 'cluster'
            cid = current.cluster

            community_mapping[(lvl, cid)].append(node_name)

            # 3. 向上递归 (parent_cluster)
            # 报错提示 'parent' 不存在，标准属性名通常是 'parent_cluster'
            # 我们做一个简单的 try-except 探测，确保万无一失
            if hasattr(current, 'parent_cluster'):
                current = current.parent_cluster
            elif hasattr(current, 'parent'):
                current = current.parent
            else:
                current = None  # 到达顶层

    print(f"   -> Generated {len(community_mapping)} communities across levels.")

    # 4. 写入 Neo4j
    print("4. Writing to Neo4j...")
    with handler.driver.session() as session:
        # 清理旧数据
        session.run("MATCH (c:Community) DETACH DELETE c")
        session.run("MATCH (e:Entity) REMOVE e.community_ids")

        cypher_write = """
        UNWIND $batch as row
        MERGE (c:Community {id: row.cid})
        SET c.level = row.level

        WITH c, row
        MATCH (e:Entity {name: row.node})
        MERGE (c)-[:HAS_MEMBER]->(e)
        """

        batch = []
        for (lvl, cid), nodes in community_mapping.items():
            for node in nodes:
                # 构造数据
                batch.append({"cid": int(cid), "level": int(lvl), "node": node})

                if len(batch) >= 1000:
                    session.run(cypher_write, batch=batch)
                    batch = []
        if batch:
            session.run(cypher_write, batch=batch)

    print("✅ 层级社区构建完成！")
    handler.close()


if __name__ == "__main__":
    run_hierarchical_community_detection()
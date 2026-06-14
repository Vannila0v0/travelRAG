import sys
import os
import networkx as nx
from collections import defaultdict
import logging

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from core.neo4j_handler import Neo4jHandler
from core.config import neo4j_password, neo4j_url, neo4j_user

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

NEO4J_URI = neo4j_url
NEO4J_USER = neo4j_user
NEO4J_PASSWORD = neo4j_password


def run_hierarchical_community_detection():
    logger.info("=== Step 4: 强力层级社区发现 (ID Lookup Fix) ===")

    handler = Neo4jHandler(NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD)

    # 1. 加载图
    logger.info("1. 从 Neo4j 加载图谱...")
    G = nx.Graph()
    cypher_load = "MATCH (s:Entity)-[r]->(t:Entity) RETURN s.name AS src, t.name AS dst"
    with handler.driver.session() as session:
        result = session.run(cypher_load)
        for record in result:
            G.add_edge(record['src'], record['dst'])

    if G.number_of_nodes() == 0:
        logger.error("❌ 图为空！")
        return

    # 2. 运行 Graspologic 分层 Leiden
    logger.info("2. 运行 Hierarchical Leiden...")
    try:
        from graspologic.partition import hierarchical_leiden

        # 强制把社区切得更细
        hierarchy = hierarchical_leiden(G, max_cluster_size=5)

    except ImportError:
        logger.error("❌ 请安装: pip install graspologic")
        return

    # 3. 解析层级结构 (ID Lookup 模式)
    logger.info("3. 解析层级结构...")

    community_mapping = defaultdict(lambda: defaultdict(list))

    # === 关键修复 Step A: 建立 ID -> 对象 的查找表 ===
    # 因为 parent 属性可能只是一个 ID，我们需要能通过 ID 找回对象
    cluster_lookup = {}

    for item in hierarchy:
        # 尝试获取该对象的 ID (兼容 cluster 或 cluster_id)
        c_id = getattr(item, "cluster", getattr(item, "cluster_id", None))
        if c_id is not None:
            cluster_lookup[c_id] = item

    # === 关键修复 Step B: 遍历并查表 ===
    for cluster_obj in hierarchy:
        # 只处理叶子节点 (Entity)
        if not hasattr(cluster_obj, 'node') or cluster_obj.node is None:
            continue

        node_name = cluster_obj.node
        current = cluster_obj

        # 开始向上递归
        while current:
            # 获取当前对象的属性
            lvl = getattr(current, "level", None)
            cid = getattr(current, "cluster", getattr(current, "cluster_id", None))

            # 安全检查：如果缺少必要属性，停止当前链条
            if lvl is None or cid is None:
                break

            community_mapping[lvl][cid].append(node_name)

            # 获取父级引用 (可能是一个 ID，也可能是一个对象)
            parent_ref = getattr(current, "parent_cluster", getattr(current, "parent", None))

            if parent_ref is None:
                current = None  # 到顶了
            elif isinstance(parent_ref, int):
                # !!! 关键点：如果是 ID (int)，去查找表里找对象 !!!
                current = cluster_lookup.get(parent_ref)
            else:
                # 如果是对象，直接用
                current = parent_ref

    found_levels = sorted(community_mapping.keys())
    logger.info(f"   -> 成功生成层级: {found_levels}")

    if not found_levels:
        logger.warning("⚠️ 未解析出任何层级，请检查 graspologic 版本或数据连通性。")
        return

    # 4. 写入 Neo4j
    logger.info("4. 写入 Neo4j...")

    with handler.driver.session() as session:
        # 清理旧数据
        session.run("MATCH (c:Community) DETACH DELETE c")
        session.run("MATCH (e:Entity) REMOVE e.community_ids")

        cypher_write_nodes = """
        UNWIND $batch as row
        MERGE (c:Community {id: row.full_id})
        SET c.level = row.level, 
            c.original_id = row.cid,
            c.title = '待生成'

        WITH c, row
        MATCH (e:Entity {name: row.node})
        MERGE (c)-[:HAS_MEMBER]->(e)
        """

        batch = []
        for lvl in found_levels:
            for cid, nodes in community_mapping[lvl].items():
                full_id = f"{lvl}_{cid}"
                for node in nodes:
                    batch.append({
                        "full_id": full_id,
                        "level": lvl,
                        "cid": cid,
                        "node": node
                    })
                    if len(batch) >= 1000:
                        session.run(cypher_write_nodes, batch=batch)
                        batch = []
        if batch:
            session.run(cypher_write_nodes, batch=batch)

        # 建立社区层级关联 (IN_COMMUNITY)
        logger.info("   -> 建立社区层级关联 (IN_COMMUNITY)...")
        # Level 0 -> Level 1
        session.run("""
            MATCH (c1:Community)-[:HAS_MEMBER]->(e:Entity)<-[:HAS_MEMBER]-(c2:Community)
            WHERE c1.level = 0 AND c2.level = 1
            MERGE (c1)-[:IN_COMMUNITY]->(c2)
        """)
        # Level 1 -> Level 2
        session.run("""
            MATCH (c1:Community)-[:IN_COMMUNITY]->(c2:Community),
                  (c2)-[:HAS_MEMBER]->(e:Entity)<-[:HAS_MEMBER]-(c3:Community)
            WHERE c2.level = 1 AND c3.level = 2
            MERGE (c2)-[:IN_COMMUNITY]->(c3)
        """)

    logger.info("✅ 强力层级社区构建完成！")
    handler.close()


if __name__ == "__main__":
    run_hierarchical_community_detection()

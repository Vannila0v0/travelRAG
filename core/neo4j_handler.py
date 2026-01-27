from neo4j import GraphDatabase
import logging

# 配置日志，方便调试
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class Neo4jHandler:
    def __init__(self, uri, user, password):
        try:
            self.driver = GraphDatabase.driver(uri, auth=(user, password))
            # 验证连接是否成功
            self.driver.verify_connectivity()
            logger.info("成功连接到 Neo4j 数据库")
        except Exception as e:
            logger.error(f"无法连接到 Neo4j: {e}")
            raise

    def close(self):
        self.driver.close()

    def add_graph_data(self, entities, relationships):
        """
        [升级版] 批量写入实体和关系，包含描述信息
        """
        with self.driver.session() as session:
            # 1. 写入实体 (使用 MERGE，如果存在则更新描述)
            # 注意：真实的 GraphRAG 会对描述做摘要合并，这里简化为覆盖或追加
            for entity in entities:
                cypher_entity = """
                MERGE (n:Entity {name: $name})
                ON CREATE SET n.type = $type, n.description = $description
                ON MATCH SET n.description = 
                    CASE 
                        WHEN size(n.description) < size($description) THEN $description 
                        ELSE n.description 
                    END
                """
                session.run(cypher_entity,
                            name=entity.name,
                            type=entity.type,
                            description=entity.description)

            # 2. 写入关系
            for rel in relationships:
                # 简单的关系名清洗
                clean_rel_type = rel.relation_type.replace(" ", "_").upper()

                cypher_rel = f"""
                MATCH (s:Entity {{name: $source}})
                MATCH (t:Entity {{name: $target}})
                MERGE (s)-[r:`{clean_rel_type}`]->(t)
                SET r.description = $description
                """
                session.run(cypher_rel,
                            source=rel.source,
                            target=rel.target,
                            description=rel.description)


    def query_1hop_neighbors(self, entity_name, limit=5):
        """
        查询一跳邻居：用于 RAG 的查询扩展
        """
        cypher = """
        MATCH (n:Entity {name: $name})-[r]-(neighbor)
        RETURN neighbor.name AS name, r.type AS rel
        LIMIT $limit
        """
        with self.driver.session() as session:
            result = session.run(cypher, name=entity_name, limit=limit)
            return [record["name"] for record in result]

        # core/neo4j_handler.py (只需在类里添加这个新方法)

    def get_local_context(self, entities: list, limit: int = 20):
        """
        [参照 graph-rag-agent 逻辑]
        根据实体列表，获取相关的子图信息（包含实体描述、关系描述）。
        逻辑：查询 (StartNode)-[Relation]->(EndNode)，提取所有描述信息拼成文本。
        """
        if not entities:
            return ""

        # Cypher 查询：
        # 1. 匹配输入实体作为起点或终点
        # 2. 抓取它们的一跳关系
        # 3. 返回三元组及其所有属性
        cypher = """
        MATCH (s:Entity)-[r]-(t:Entity)
        WHERE s.name IN $names
        WITH s, r, t
        LIMIT $limit
        RETURN s.name, s.description, type(r), r.description, t.name, t.description
        """

        context_lines = []
        with self.driver.session() as session:
            result = session.run(cypher, names=entities, limit=limit)
            for record in result:
                # 容错处理：防止 description 为 None
                s_desc = f"({record['s.description']})" if record['s.description'] else ""
                t_desc = f"({record['t.description']})" if record['t.description'] else ""

                # 关系描述处理：如果有描述就显示描述，没有就显示类型
                r_info = record['type(r)']
                if record['r.description']:
                    r_info += f": {record['r.description']}"

                # 格式化为自然语言风格，方便 LLM 阅读
                # 示例：实体[夜游成人票(适用于1.5米以上...)] --[价格: 门市价]--> 实体[210元(金额)]
                line = f"实体[{record['s.name']}{s_desc}] --[{r_info}]--> 实体[{record['t.name']}{t_desc}]"
                context_lines.append(line)

        return "\n".join(context_lines)


    def fuzzy_search(self, keyword):
        """
        模糊查找实体名（防止用户输错字）
        """
        cypher = """
        MATCH (n:Entity)
        WHERE n.name CONTAINS $keyword
        RETURN n.name AS name
        LIMIT 1
        """
        with self.driver.session() as session:
            result = session.run(cypher, keyword=keyword)
            record = result.single()
            return record["name"] if record else None

    # ... (上面的代码保持不变) ...

    def perform_dqa(self):
        """
        执行数据质量治理 (DQA - Data Quality Assurance)
        清洗脏数据、归一化关系、移除孤立点
        """
        logger.info("开始执行 DQA 数据治理流程...")

        with self.driver.session() as session:
            # 1. 【拓扑检查】移除自环 (Self-loops)
            # 解释：删除节点指向自己的边 (例如：外滩 -> 外滩)
            res1 = session.run("MATCH (n)-[r]->(n) DELETE r RETURN count(r) as count")
            count1 = res1.single()["count"]
            logger.info(f"DQA: 已移除 {count1} 条自环边")

            # 2. 【拓扑检查】移除孤立节点 (Orphan Nodes)
            # 解释：删除没有任何边连入或连出的节点
            res2 = session.run("""
MATCH (n)
WITH n, COUNT { (n)--() } AS degree
WHERE degree = 0
DETACH DELETE n
RETURN count(n) AS count
""")
            count2 = res2.single()["count"]
            logger.info(f"DQA: 已移除 {count2} 个孤立节点")

            # 3. 【Schema归一化】统一关系名称 (Normalization)
            # 解释：将各种乱七八糟的“位于”表达统一为 "LOCATED_IN"
            # 注意：这里你可以根据实际观察到的脏数据添加更多规则
            normalization_map = {
                "位于": "LOCATED_IN",
                "is_in": "LOCATED_IN",
                "in": "LOCATED_IN",
                "belongs_to": "LOCATED_IN",
                "毗邻": "NEAR",
                "near": "NEAR",
                "nearby": "NEAR"
            }

            total_normalized = 0
            for dirty_rel, clean_rel in normalization_map.items():
                # 查找旧关系，建立新关系，删除旧关系
                cypher = f"""
                MATCH (s)-[r:`{dirty_rel}`]->(t)
                MERGE (s)-[:{clean_rel}]->(t)
                DELETE r
                RETURN count(r) as count
                """
                # 因为可能根本没有这个脏关系，所以要加 try-except 防止报错
                try:
                    res = session.run(cypher)
                    count = res.single()["count"]
                    if count > 0:
                        logger.info(f"DQA: 将 {count} 条 '{dirty_rel}' 归一化为 '{clean_rel}'")
                        total_normalized += count
                except Exception:
                    # 如果库里没有 '位于' 这种关系，Neo4j 可能会报错或单纯返回0，忽略即可
                    pass

            logger.info(f"DQA: 关系归一化完成，共处理 {total_normalized} 条关系")



        logger.info("DQA 流程执行完毕！图谱已清洗。")

# 初始化一个全局实例
# 如果你的 docker 密码改了，这里记得改
neo4j_client = Neo4jHandler("bolt://localhost:7687", "neo4j", "password123")
import json
import logging

from neo4j import GraphDatabase

from core.config import neo4j_password, neo4j_url, neo4j_user


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class Neo4jHandler:
    def __init__(self, uri, user, password):
        try:
            self.driver = GraphDatabase.driver(uri, auth=(user, password))
            self.driver.verify_connectivity()
            logger.info("Connected to Neo4j")
        except Exception as exc:
            logger.error(f"Failed to connect to Neo4j: {exc}")
            raise

    def close(self):
        self.driver.close()

    def execute_query(self, query, parameters=None):
        with self.driver.session() as session:
            return list(session.run(query, parameters or {}))

    @staticmethod
    def _clean_rel_type(rel_type):
        clean_rel_type = str(rel_type or "RELATED_TO").replace(" ", "_").replace("`", "").upper()
        return clean_rel_type or "RELATED_TO"

    @staticmethod
    def _dump_source_refs(source_refs):
        values = []
        for ref in source_refs or []:
            if hasattr(ref, "model_dump"):
                payload = ref.model_dump()
            elif hasattr(ref, "dict"):
                payload = ref.dict()
            else:
                payload = dict(ref)
            values.append(json.dumps(payload, ensure_ascii=False, sort_keys=True))
        return values

    @staticmethod
    def _source_ref_ids(source_refs):
        ids = []
        for ref in source_refs or []:
            chunk_id = getattr(ref, "chunk_id", None)
            if chunk_id is None and isinstance(ref, dict):
                chunk_id = ref.get("chunk_id")
            if chunk_id and chunk_id not in ids:
                ids.append(chunk_id)
        return ids

    def add_document_chunk(self, doc_id, doc_name, source_path, chunk_id, chunk_index, text, metadata=None):
        metadata_json = json.dumps(metadata or {}, ensure_ascii=False, sort_keys=True)
        with self.driver.session() as session:
            session.run(
                """
                MERGE (d:Document {id: $doc_id})
                SET d.name = $doc_name,
                    d.source_path = $source_path
                MERGE (c:Chunk {id: $chunk_id})
                SET c.doc_id = $doc_id,
                    c.chunk_index = $chunk_index,
                    c.text = $text,
                    c.source_path = $source_path,
                    c.metadata_json = $metadata_json
                MERGE (d)-[:HAS_CHUNK]->(c)
                """,
                doc_id=doc_id,
                doc_name=doc_name,
                source_path=source_path,
                chunk_id=chunk_id,
                chunk_index=chunk_index,
                text=text,
                metadata_json=metadata_json,
            )

    def add_graph_data(self, entities, relationships):
        with self.driver.session() as session:
            for entity in entities:
                source_ref_ids = self._source_ref_ids(getattr(entity, "source_refs", []))
                source_refs_json = self._dump_source_refs(getattr(entity, "source_refs", []))
                session.run(
                    """
                    MERGE (n:Entity {name: $name})
                    ON CREATE SET
                        n.type = $type,
                        n.description = $description,
                        n.source_ref_ids = $source_ref_ids,
                        n.source_refs_json = $source_refs_json
                    ON MATCH SET
                        n.description =
                            CASE
                                WHEN size(coalesce(n.description, "")) < size($description) THEN $description
                                ELSE n.description
                            END,
                        n.source_ref_ids = reduce(ids = coalesce(n.source_ref_ids, []), id IN $source_ref_ids | CASE WHEN id IN ids THEN ids ELSE ids + id END),
                        n.source_refs_json = reduce(refs = coalesce(n.source_refs_json, []), ref IN $source_refs_json | CASE WHEN ref IN refs THEN refs ELSE refs + ref END)
                    """,
                    name=entity.name,
                    type=entity.type,
                    description=entity.description,
                    source_ref_ids=source_ref_ids,
                    source_refs_json=source_refs_json,
                )

                for chunk_id in source_ref_ids:
                    session.run(
                        """
                        MATCH (n:Entity {name: $name})
                        MATCH (c:Chunk {id: $chunk_id})
                        MERGE (n)-[:MENTIONED_IN]->(c)
                        """,
                        name=entity.name,
                        chunk_id=chunk_id,
                    )

            for rel in relationships:
                clean_rel_type = self._clean_rel_type(rel.relation_type)
                source_ref_ids = self._source_ref_ids(getattr(rel, "source_refs", []))
                source_refs_json = self._dump_source_refs(getattr(rel, "source_refs", []))
                cypher = f"""
                MATCH (s:Entity {{name: $source}})
                MATCH (t:Entity {{name: $target}})
                MERGE (s)-[r:`{clean_rel_type}`]->(t)
                SET r.description = $description,
                    r.source_ref_ids = reduce(ids = coalesce(r.source_ref_ids, []), id IN $source_ref_ids | CASE WHEN id IN ids THEN ids ELSE ids + id END),
                    r.source_refs_json = reduce(refs = coalesce(r.source_refs_json, []), ref IN $source_refs_json | CASE WHEN ref IN refs THEN refs ELSE refs + ref END)
                """
                session.run(
                    cypher,
                    source=rel.source,
                    target=rel.target,
                    description=rel.description,
                    source_ref_ids=source_ref_ids,
                    source_refs_json=source_refs_json,
                )

    def add_triple(self, source, relation, target, description=""):
        from graph_engine.schema import Entity, Relationship

        entities = [
            Entity(name=source, type="UNKNOWN", description=""),
            Entity(name=target, type="UNKNOWN", description=""),
        ]
        relationships = [
            Relationship(
                source=source,
                target=target,
                relation_type=relation,
                description=description,
            )
        ]
        self.add_graph_data(entities, relationships)

    def query_1hop_neighbors(self, entity_name, limit=5):
        cypher = """
        MATCH (n:Entity {name: $name})-[r]-(neighbor)
        RETURN neighbor.name AS name, type(r) AS rel
        LIMIT $limit
        """
        with self.driver.session() as session:
            result = session.run(cypher, name=entity_name, limit=limit)
            return [record["name"] for record in result]

    def get_local_context(self, entities: list, limit: int = 20):
        if not entities:
            return ""

        cypher = """
        MATCH (s:Entity)-[r]-(t:Entity)
        WHERE s.name IN $names
        WITH s, r, t
        LIMIT $limit
        RETURN s.name, s.description, type(r), r.description, t.name, t.description, r.source_ref_ids
        """

        context_lines = []
        with self.driver.session() as session:
            result = session.run(cypher, names=entities, limit=limit)
            for record in result:
                s_desc = f"({record['s.description']})" if record["s.description"] else ""
                t_desc = f"({record['t.description']})" if record["t.description"] else ""
                r_info = record["type(r)"]
                if record["r.description"]:
                    r_info += f": {record['r.description']}"
                refs = record["r.source_ref_ids"] or []
                ref_text = f" source={','.join(refs)}" if refs else ""
                context_lines.append(
                    f"Entity[{record['s.name']}{s_desc}] --[{r_info}{ref_text}]--> Entity[{record['t.name']}{t_desc}]"
                )

        return "\n".join(context_lines)

    def fuzzy_search(self, keyword):
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

    def perform_dqa(self):
        logger.info("Running DQA cleanup")
        with self.driver.session() as session:
            res1 = session.run("MATCH (n)-[r]->(n) DELETE r RETURN count(r) as count")
            logger.info(f"DQA removed self-loop relationships: {res1.single()['count']}")

            res2 = session.run(
                """
                MATCH (n:Entity)
                WITH n, COUNT { (n)--() } AS degree
                WHERE degree = 0
                DETACH DELETE n
                RETURN count(n) AS count
                """
            )
            logger.info(f"DQA removed orphan entities: {res2.single()['count']}")

            normalization_map = {
                "位于": "LOCATED_IN",
                "is_in": "LOCATED_IN",
                "in": "LOCATED_IN",
                "belongs_to": "LOCATED_IN",
                "毗邻": "NEAR",
                "near": "NEAR",
                "nearby": "NEAR",
            }

            total_normalized = 0
            for dirty_rel, clean_rel in normalization_map.items():
                cypher = f"""
                MATCH (s)-[r:`{dirty_rel}`]->(t)
                MERGE (s)-[nr:{clean_rel}]->(t)
                SET nr.description = r.description,
                    nr.source_ref_ids = r.source_ref_ids,
                    nr.source_refs_json = r.source_refs_json
                DELETE r
                RETURN count(r) as count
                """
                try:
                    res = session.run(cypher)
                    count = res.single()["count"]
                    total_normalized += count
                    if count:
                        logger.info(f"DQA normalized {count} '{dirty_rel}' relationships to '{clean_rel}'")
                except Exception:
                    pass

            logger.info(f"DQA normalized relationships: {total_normalized}")


class LazyNeo4jClient:
    def __init__(self):
        self._client = None

    def _get_client(self):
        if self._client is None:
            self._client = Neo4jHandler(neo4j_url, neo4j_user, neo4j_password)
        return self._client

    def close(self):
        if self._client is not None:
            self._client.close()
            self._client = None

    def __getattr__(self, name):
        return getattr(self._get_client(), name)


neo4j_client = LazyNeo4jClient()

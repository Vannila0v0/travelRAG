CREATE CONSTRAINT entity_name_unique IF NOT EXISTS
FOR (e:Entity)
REQUIRE e.name IS UNIQUE;

CREATE INDEX entity_type_index IF NOT EXISTS
FOR (e:Entity)
ON (e.type);

CREATE CONSTRAINT community_id_unique IF NOT EXISTS
FOR (c:Community)
REQUIRE c.id IS UNIQUE;

CREATE INDEX community_level_index IF NOT EXISTS
FOR (c:Community)
ON (c.level);

CREATE INDEX community_title_index IF NOT EXISTS
FOR (c:Community)
ON (c.title);

CREATE CONSTRAINT document_id_unique IF NOT EXISTS
FOR (d:Document)
REQUIRE d.id IS UNIQUE;

CREATE CONSTRAINT chunk_id_unique IF NOT EXISTS
FOR (c:Chunk)
REQUIRE c.id IS UNIQUE;

CREATE INDEX chunk_doc_id_index IF NOT EXISTS
FOR (c:Chunk)
ON (c.doc_id);

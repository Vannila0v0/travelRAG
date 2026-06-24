import unittest

from langchain_core.documents import Document

from query_engine.vector_query import VectorQueryEngine


class FakeVectorStore:
    def __init__(self, results):
        self.results = results
        self.requested_k = None

    def similarity_search_with_score(self, query, k):
        self.requested_k = k
        return self.results[:k]


class VectorRerankTests(unittest.TestCase):
    def test_entity_match_beats_generic_hot_spring_ticket_chunk(self):
        generic = Document(
            page_content="贺州温泉旅游有限责任公司温泉门票价格，有老人优惠和儿童半价政策。",
            metadata={"chunk_id": "generic", "file_name": "hezhou.docx"},
        )
        longsheng = Document(
            page_content="龙胜温泉门票对学生、60岁以上老人、现役军人和退役军人有9折优惠。",
            metadata={"chunk_id": "longsheng", "file_name": "tourism_dpo.md"},
        )
        engine = VectorQueryEngine(llm=object())
        engine._vectorstore = FakeVectorStore([(generic, 0.68), (longsheng, 0.72)])

        docs, sources = engine.retrieve("龙胜温泉门票有哪些优惠政策？", k=1)

        self.assertEqual(docs[0].metadata["chunk_id"], "longsheng")
        self.assertEqual(sources[0].chunk_id, "longsheng")
        self.assertEqual(engine._vectorstore.requested_k, 4)


if __name__ == "__main__":
    unittest.main()

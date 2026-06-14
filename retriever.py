# retriever.py

from collections import OrderedDict

def retrieve_with_multi_query(retriever, queries: list[str]):
    all_docs = []

    for q in queries:
        docs = retriever.get_relevant_documents(q)
        all_docs.extend(docs)

    return dedup_docs(all_docs)

def dedup_docs(docs):
    unique = OrderedDict()
    for doc in docs:
        unique[doc.page_content] = doc
    return list(unique.values())

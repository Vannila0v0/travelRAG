# run_demo.py

from vectorstore import build_vectorstore_from_file
from rag_pipeline import rag_answer
from config import TOP_K

if __name__ == "__main__":
    vectorstore = build_vectorstore_from_file("data/旅游问答.md")
    retriever = vectorstore.as_retriever(search_kwargs={"k": TOP_K})

    while True:
        q = input("请输入问题：")
        if q.lower() in ("exit", "quit"):
            break

        answer = rag_answer(q, retriever)
        print("\nAI 回答：")
        print(answer)
        print("=" * 50)

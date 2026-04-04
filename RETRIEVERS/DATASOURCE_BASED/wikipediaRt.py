from langchain_community.retrievers import WikipediaRetriever

# ✅ Create retriever
retriever = WikipediaRetriever(
    top_k_results=2,   # kitne docs chahiye
    lang="en"
)

# ✅ Query
query = "Virat Kohli"

docs = retriever.invoke(query)

# ✅ Print results
for i, doc in enumerate(docs):
    print(f"\n--- Result {i+1} ---\n")
    print(doc.page_content[:500]) 
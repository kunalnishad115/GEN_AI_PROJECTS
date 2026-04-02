import os
from dotenv import load_dotenv

from langchain_cohere import CohereEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document

load_dotenv()

# ✅ Cohere Embeddings
embeddings = CohereEmbeddings(
    cohere_api_key=os.getenv("COHERE_KEY"),
    model="embed-english-v3.0"
)

# ✅ Create Vector Store
vector_store = Chroma(
    collection_name="ipl_players",
    embedding_function=embeddings,
    persist_directory="./chroma_db"   # optional (for saving)
)

# ✅ Documents
doc1 = Document(
    page_content="Virat Kohli is one of the most successful and consistent batsmen in IPL history...",
    metadata={"team": "Royal Challengers Bangalore"}
)

doc2 = Document(
    page_content="Rohit Sharma is the most successful captain in IPL history...",
    metadata={"team": "Mumbai Indians"}
)

doc3 = Document(
    page_content="MS Dhoni, famously known as Captain Cool...",
    metadata={"team": "Chennai Super Kings"}
)

doc4 = Document(
    page_content="Jasprit Bumrah is considered one of the best fast bowlers...",
    metadata={"team": "Mumbai Indians"}
)

doc5 = Document(
    page_content="Ravindra Jadeja is a dynamic all-rounder...",
    metadata={"team": "Chennai Super Kings"}
)

docs = [doc1, doc2, doc3, doc4, doc5]

# ✅ Add documents
vector_store.add_documents(docs)

# ✅ Persist DB (important in VS Code)
vector_store.persist()

# ✅ View documents
print(vector_store.get(include=['documents', 'metadatas']))

# ✅ Similarity search
print("\n🔍 Similarity Search:")
results = vector_store.similarity_search(
    query='Who among these are a bowler?',
    k=2
)
for r in results:
    print(r.page_content)

# ✅ With score
print("\n📊 Similarity with score:")
results_with_score = vector_store.similarity_search_with_score(
    query='Who among these are a bowler?',
    k=2
)
for r, score in results_with_score:
    print(score, "→", r.page_content)

# ✅ Filter search
print("\n🎯 Filter Search (CSK):")
filtered = vector_store.similarity_search(
    query="",
    filter={"team": "Chennai Super Kings"}
)
for r in filtered:
    print(r.page_content)

# ❗ Update document (NOTE: need ID)
# First get IDs
data = vector_store.get()
doc_ids = data["ids"]

print("\nDoc IDs:", doc_ids)

# Example update (use actual ID from above)
updated_doc = Document(
    page_content="Virat Kohli is the highest run scorer in IPL history...",
    metadata={"team": "Royal Challengers Bangalore"}
)

vector_store.update_document(
    document_id=doc_ids[0],  # use real ID
    document=updated_doc
)
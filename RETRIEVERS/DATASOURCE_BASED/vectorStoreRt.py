import os
from dotenv import load_dotenv
from langchain_core.documents import Document
from langchain_cohere import CohereEmbeddings
from langchain_community.vectorstores import Chroma

load_dotenv()

# ✅ Embeddings
embeddings = CohereEmbeddings(
    cohere_api_key=os.getenv("COHERE_KEY"),
    model="embed-english-v3.0"
)

# ✅ Sample Documents
docs = [
    Document(page_content="Virat Kohli is a top batsman from RCB.", metadata={"team": "RCB"}),
    Document(page_content="Rohit Sharma leads Mumbai Indians and is a great opener.", metadata={"team": "MI"}),
    Document(page_content="MS Dhoni is a legendary finisher and captain of CSK.", metadata={"team": "CSK"}),
    Document(page_content="Jasprit Bumrah is a fast bowler known for yorkers.", metadata={"team": "MI"}),
    Document(page_content="Ravindra Jadeja is an all-rounder from CSK.", metadata={"team": "CSK"})
]

# ✅ Create Vector Store
vector_store = Chroma.from_documents(
    documents=docs,
    embedding=embeddings,
    collection_name="ipl_data",
    persist_directory="./chroma_db"
)

# ✅ Convert to Retriever
retriever = vector_store.as_retriever(
    search_type="similarity",
    search_kwargs={"k": 2}
)

# ✅ Query
query = "Who is a bowler?"

results = retriever.invoke(query)

# ✅ Output
for i, doc in enumerate(results):
    print(f"\n--- Result {i+1} ---\n")
    print(doc.page_content)
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

# ✅ Documents
docs = [
    Document(page_content="Virat Kohli is a top batsman from RCB.", metadata={"team": "RCB"}),
    Document(page_content="Rohit Sharma is a successful captain and opener.", metadata={"team": "MI"}),
    Document(page_content="MS Dhoni is a legendary finisher and captain.", metadata={"team": "CSK"}),
    Document(page_content="Jasprit Bumrah is a fast bowler known for yorkers.", metadata={"team": "MI"}),
    Document(page_content="Ravindra Jadeja is an all-rounder with strong bowling skills.", metadata={"team": "CSK"}),
    Document(page_content="Mohammed Shami is a fast bowler with swing.", metadata={"team": "GT"})
]

# ✅ Vector Store
vector_store = Chroma.from_documents(
    documents=docs,
    embedding=embeddings,
    collection_name="ipl_mmr",
    persist_directory="./chroma_db"
)

# ✅ MMR Retriever
retriever = vector_store.as_retriever(
    search_type="mmr",
    search_kwargs={
        "k": 3,          # final results
        "fetch_k": 10,   # initial pool
        "lambda_mult": 0.5
    }
)

# ✅ Query
query = "Who are bowlers?"

results = retriever.invoke(query)

# ✅ Output
for i, doc in enumerate(results):
    print(f"\n--- Result {i+1} ---\n")
    print(doc.page_content)
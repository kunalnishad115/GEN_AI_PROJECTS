import os
from dotenv import load_dotenv

from langchain_core.documents import Document
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from langchain_cohere import CohereEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.retrievers.multi_query import MultiQueryRetriever

load_dotenv()

# ✅ LLM (Groq)
llm = ChatOpenAI(
    model="llama-3.1-8b-instant",
    openai_api_key=os.getenv("GROQ_KEY"),
    openai_api_base="https://api.groq.com/openai/v1"
)

# ✅ Embeddings (Cohere)
embeddings = CohereEmbeddings(
    cohere_api_key=os.getenv("COHERE_API_KEY"),
    model="embed-english-v3.0"
)

# ✅ Documents
docs = [
    Document(page_content="Virat Kohli is a top batsman.", metadata={"team": "RCB"}),
    Document(page_content="Jasprit Bumrah is a fast bowler.", metadata={"team": "MI"}),
    Document(page_content="Mohammed Shami is a swing bowler.", metadata={"team": "GT"}),
    Document(page_content="Ravindra Jadeja is an all-rounder.", metadata={"team": "CSK"}),
    Document(page_content="MS Dhoni is a finisher and captain.", metadata={"team": "CSK"})
]

# ✅ Vector Store
vector_store = Chroma.from_documents(
    documents=docs,
    embedding=embeddings,
    collection_name="multi_query_demo",
    persist_directory="./chroma_db"
)

# ✅ Base Retriever
base_retriever = vector_store.as_retriever(search_kwargs={"k": 2})

# ✅ MultiQuery Retriever
retriever = MultiQueryRetriever.from_llm(
    retriever=base_retriever,
    llm=llm
)

# ✅ Query
query = "Who are bowlers?"

results = retriever.get_relevant_documents(query)

# ✅ Output
for i, doc in enumerate(results):
    print(f"\n--- Result {i+1} ---\n")
    print(doc.page_content)
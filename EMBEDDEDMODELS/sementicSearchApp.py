from langchain_cohere import CohereEmbeddings
from dotenv import load_dotenv
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import os

load_dotenv()

# ✅ Cohere embeddings (no dimensions param)
embedding = CohereEmbeddings(
    cohere_api_key=os.getenv("COHERE_KEY"),
    model="embed-english-v3.0"
)

documents = [
    "Virat Kohli is an Indian cricketer known for his aggressive batting and leadership.",
    "MS Dhoni is a former Indian captain famous for his calm demeanor and finishing skills.",
    "Sachin Tendulkar, also known as the 'God of Cricket', holds many batting records.",
    "Rohit Sharma is known for his elegant batting and record-breaking double centuries.",
    "Jasprit Bumrah is an Indian fast bowler known for his unorthodox action and yorkers."
]

query = 'tell me about bumrah'

# ✅ Generate embeddings
doc_embeddings = embedding.embed_documents(documents)
query_embedding = embedding.embed_query(query)

# ✅ Similarity
scores = cosine_similarity([query_embedding], doc_embeddings)[0]

index, score = sorted(
    list(enumerate(scores)),
    key=lambda x: x[1]
)[-1]

print("Query:", query)
print("Most Relevant Doc:", documents[index])
print("Similarity Score:", score)
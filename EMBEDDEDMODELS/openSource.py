# from langchain_cohere import CohereEmbeddings
# from dotenv import load_dotenv
# import os

# # Load env
# load_dotenv()

# # Initialize embeddings
# embeddings = CohereEmbeddings(
#     cohere_api_key=os.getenv("COHERE_KEY"),
#     model="embed-english-v3.0"
#     # dimensions=32
# )

# # Example query embedding
# documents=[
#     "Artificial Intelligence (AI) is the simulation of human intelligence processes by machines, especially computer systems", 
#     "These processes include learning (the acquisition of information and rules for using the information), reasoning (using rules to reach approximate or definite conclusions), and self-correction.",
#     " AI can be categorized into narrow AI, which is designed to perform a narrow task (like facial recognition" 
# ]
# query = "What is Artificial Intelligence?"

# vector = embeddings.embed_documents(documents)

# print(vector[:3])  # print first 3 values
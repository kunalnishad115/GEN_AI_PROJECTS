from langchain_huggingface import HuggingFaceEndpoint
from dotenv import load_dotenv
import os

load_dotenv()

llm = HuggingFaceEndpoint(
    repo_id="google/flan-t5-large",
    huggingfacehub_api_token=os.getenv("HUGGINGFACE_ACCESS_TOKEN"),
    task="text-generation",
    max_new_tokens=100,
    temperature=1.0
)

result = llm.invoke("What is the capital of India")

print(result)
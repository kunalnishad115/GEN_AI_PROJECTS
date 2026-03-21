from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
import os

load_dotenv()

chatModel = ChatOpenAI(
    model="llama-3.1-8b-instant",
    openai_api_key=os.getenv("GROQ_KEY"),
    openai_api_base="https://api.groq.com/openai/v1",
    temperature=1.0,
    max_completion_tokens=100
)

response = chatModel.invoke("give me a joke for a student name is Rahul Pathak ?")

print(response.content)
# print(response)
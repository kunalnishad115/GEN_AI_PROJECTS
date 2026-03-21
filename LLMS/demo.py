from langchain_openai import ChatOpenAI 
from dotenv import load_dotenv
import os

load_dotenv()

llm = ChatOpenAI(
    model="llama-3.1-8b-instant",
    openai_api_key=os.getenv("GEMINI_KEY"),
    openai_api_base="https://api.groq.com/openai/v1"
)

response = llm.invoke("when i add the 90+90 what i got ?")

print(response.content)
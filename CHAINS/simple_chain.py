from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
import os

load_dotenv()


prompt=PromptTemplate(
  template='Tell me The Powerfull fact about the Army/defence of Perticular {place}',
  input_variables=['place']
)

# ✅ Groq Model (via OpenAI-compatible API)
model = ChatOpenAI(
    model="llama-3.1-8b-instant",
    openai_api_key=os.getenv("GROQ_KEY"),
    openai_api_base="https://api.groq.com/openai/v1"
)

parser=StrOutputParser()

chain=prompt | model | parser

result=chain.invoke({'place': 'India'})

print(result)

# chain.get_graph().draw_ascii()  for visualizing the chain structure in ASCII format (optional) 
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
import os

load_dotenv()

# ✅ Groq Model (via OpenAI-compatible API)
model = ChatOpenAI(
    model="llama-3.1-8b-instant",
    openai_api_key=os.getenv("GROQ_KEY"),
    openai_api_base="https://api.groq.com/openai/v1"
)

prompt1=PromptTemplate(
  template='Genarate The Full Documented Report File about the {topic}',
  input_variables=['topic']
)

prompt2=PromptTemplate(
  template='Summarize The Perticular Report File in 3 sort lines by Use the following {text}',
  input_variables=['text']
)


parser=StrOutputParser()

chain = prompt1 | model | parser | prompt2 | model | parser

result=chain.invoke({'topic': 'India Defence'})

print(result)


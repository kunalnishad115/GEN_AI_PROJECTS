from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableSequence
import os

load_dotenv()


prompt1=PromptTemplate(
  template='Tell me The importance to learn The following {topic}',
  input_variables=['topic']
)

# ✅ Groq Model (via OpenAI-compatible API)
model = ChatOpenAI(
    model="llama-3.1-8b-instant",
    openai_api_key=os.getenv("GROQ_KEY"),
    openai_api_base="https://api.groq.com/openai/v1"
)

prompt2=PromptTemplate(
  template='Summarize The Perticular Report File in 3 sort lines by Use the following {text}',
  input_variables=['text']
)

parser=StrOutputParser()


chain=RunnableSequence(prompt1, model , parser , prompt2 , model , parser)

result=chain.invoke({'topic': 'Gen-Ai'})

print(result)



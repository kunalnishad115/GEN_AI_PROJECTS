from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableSequence , RunnableParallel
import os

load_dotenv()


prompt1=PromptTemplate(
  template='write the sort tweet about  {topic} max length should be 1 line',
  input_variables=['topic']
)

# ✅ Groq Model (via OpenAI-compatible API)
model = ChatOpenAI(
    model="llama-3.1-8b-instant",
    openai_api_key=os.getenv("GROQ_KEY"),
    openai_api_base="https://api.groq.com/openai/v1"
)

prompt2=PromptTemplate(
  template='write the sort linkedin post about {topic} max length should be 1 line',
  input_variables=['topic']
)

parser=StrOutputParser()

parallel_chain=RunnableParallel(
  {
    'tweet' : RunnableSequence(prompt1, model , parser),
    'linkedin': RunnableSequence(prompt2, model , parser)
  }
)

result=parallel_chain.invoke({'topic': 'Gen-Ai'})

print(result)
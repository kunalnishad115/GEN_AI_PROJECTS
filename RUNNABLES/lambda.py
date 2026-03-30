from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableSequence , RunnableParallel , RunnablePassthrough , RunnableLambda
import os

load_dotenv()


prompt=PromptTemplate(
  template='Tell me The importance to learn The following {topic} max Length should be 1 line',
  input_variables=['topic']
)

def text_len(topic):
  return len(topic.split())

# ✅ Groq Model (via OpenAI-compatible API)
model = ChatOpenAI(
    model="llama-3.1-8b-instant",
    openai_api_key=os.getenv("GROQ_KEY"),
    openai_api_base="https://api.groq.com/openai/v1"
)

parser=StrOutputParser()


topic_chain=RunnableSequence(prompt, model, parser)

parallel_chain=RunnableParallel(
  {
    'topic':RunnablePassthrough(),
    'length':RunnableLambda(text_len)
  }
)

final_chain=RunnableSequence(topic_chain , parallel_chain)

result=final_chain.invoke({'topic': 'Gen-Ai'})

print(result)



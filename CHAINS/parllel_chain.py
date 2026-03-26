from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableParallel
import os

load_dotenv()

# ✅ Groq Model (via OpenAI-compatible API)
model_notes = ChatOpenAI(
    model="llama-3.1-8b-instant",
    openai_api_key=os.getenv("GROQ_KEY"),
    openai_api_base="https://api.groq.com/openai/v1"
)

model_quiz = ChatOpenAI(
    model="llama-3.1-8b-instant",
    openai_api_key=os.getenv("GROQ_KEY"),
    openai_api_base="https://api.groq.com/openai/v1"
)

prompt_notes=PromptTemplate(
  template='Genrate The short Notes about the {text}',
  input_variables=['text']
)

prompt_quiz=PromptTemplate(
  template='Genarate The 5 Quiz Questions about the {text}',
  input_variables=['text']
)

prompt_merged = PromptTemplate(
  template="""
Merge the following Notes and Quiz Questions into a single well-structured document:

Notes:
{notes}

Quiz Questions:
{quiz}
""",
  input_variables=['notes', 'quiz']
)

parser=StrOutputParser()

parllel_chain=RunnableParallel(
  {
    'notes' : prompt_notes | model_notes | parser,
    'quiz' : prompt_quiz | model_quiz | parser
  }
)

merged_chain =prompt_merged | model_quiz | parser

chain=parllel_chain | merged_chain

text="""
The Indian Army is the land-based branch of the Indian Armed Forces. It is responsible for safeguarding the nation's borders and maintaining internal security. The Indian Army has a rich history, dating back to its formation in 1895. It has played a crucial role in various conflicts, including the Indo-Pakistani wars and the Kargil War. The army is known for its professionalism, discipline, and dedication to duty. It operates in diverse terrains, from the high-altitude regions of Ladakh to the dense forests of the Northeast. The Indian Army also contributes to international peacekeeping missions and disaster relief efforts. With a strength of over 1.4 million active personnel, it is one of the largest standing armies in the world.
"""
result=chain.invoke({'text': text})

print(result)
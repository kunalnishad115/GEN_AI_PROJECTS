import os
from dotenv import load_dotenv

from langchain_community.document_loaders import TextLoader
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

# ✅ Load file (static path)
loader = TextLoader("DOCUMENTLOADERS/jokes.txt", encoding="utf-8")
documents = loader.load()

text_data = "\n".join([doc.page_content for doc in documents])

# ✅ Your Groq model (OpenAI-compatible)
llm = ChatOpenAI(
    model="llama-3.1-8b-instant",
    openai_api_key=os.getenv("GROQ_KEY"),
    openai_api_base="https://api.groq.com/openai/v1",
    temperature=0.3
)

# ✅ Prompt (LCEL style)
prompt = ChatPromptTemplate.from_template("""
Summarize the following jokes dataset:
- overall theme
- humor type
- keep it concise

TEXT:
{text}
""")


parser= StrOutputParser()
# ✅ LCEL Chain
chain = prompt | llm | parser

# ✅ Run
response = chain.invoke({"text": text_data})

print("\n🔥 SUMMARY:\n")
print(response)

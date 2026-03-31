import os
from dotenv import load_dotenv

from langchain_community.document_loaders import WebBaseLoader
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

# ✅ Load web page
url = "https://www.amazon.in/avvatar-PERFORMANCE-PROTEIN-Flavour-Servings/dp/B0DSFZL1WZ"
loader = WebBaseLoader(url)

documents = loader.load()

# Combine content
text_data = "\n".join([doc.page_content for doc in documents])

# ✅ Your GROQ model
llm = ChatOpenAI(
    model="llama-3.1-8b-instant",
    openai_api_key=os.getenv("GROQ_KEY"),
    openai_api_base="https://api.groq.com/openai/v1",
    temperature=0
)

# ✅ Prompt (QA style)
prompt = ChatPromptTemplate.from_template("""
You are a product analysis assistant.

Based ONLY on the given content, answer the question.
If the answer is not clearly present, say "Not mentioned".

CONTENT:
{text}

QUESTION:
{question}
""")

# ✅ LCEL chain
chain = prompt | llm | StrOutputParser()

# ✅ Ask question
response = chain.invoke({
    "text": text_data,
    "question": "Is this whey protein isolate or concentrate?"
})

print("\n🔥 ANSWER:\n")
print(response)
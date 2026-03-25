from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser
import os

load_dotenv()

# ✅ Groq Model (via OpenAI-compatible API)
model = ChatOpenAI(
    model="llama-3.1-8b-instant",
    openai_api_key=os.getenv("GROQ_KEY"),
    openai_api_base="https://api.groq.com/openai/v1"
)

# ✅ Parser
parser = JsonOutputParser()

# ✅ Prompt Template
template = PromptTemplate(
    template="Give me 1 facts about {topic}\n{format_instruction}",
    input_variables=["topic"],
    partial_variables={
        "format_instruction": parser.get_format_instructions()
    }
)

# ✅ LCEL Chain
chain = template | model | parser

# ✅ Invoke
result = chain.invoke({"topic": "black hole"})

print(result)
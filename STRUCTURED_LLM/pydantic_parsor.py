from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field
from typing import List
from dotenv import load_dotenv
import os

load_dotenv()

# ✅ Model (Groq)
model = ChatOpenAI(
    model="llama-3.1-8b-instant",
    openai_api_key=os.getenv("GROQ_KEY"),
    openai_api_base="https://api.groq.com/openai/v1"
)

# ✅ Schema
class Facts(BaseModel):
    facts: List[str] = Field(description="List of 5 facts about the topic")

# ✅ Parser
parser = PydanticOutputParser(pydantic_object=Facts)

# ✅ Prompt
template = PromptTemplate(
    template="Give me 5 facts about {topic}.\n{format_instructions}",
    input_variables=["topic"],
    partial_variables={
        "format_instructions": parser.get_format_instructions()
    }
)

# ✅ Chain
chain = template | model | parser

# ✅ Run
result = chain.invoke({"topic": "black hole"})

print(result)
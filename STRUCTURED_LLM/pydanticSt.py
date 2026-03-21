from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field
from typing import List, Optional, Literal
from dotenv import load_dotenv
import os

load_dotenv()

# ✅ LLM (Groq)
model = ChatOpenAI(
    model="llama-3.1-8b-instant",
    openai_api_key=os.getenv("GROQ_KEY"),
    openai_api_base="https://api.groq.com/openai/v1"
)

# ✅ Pydantic Schema
class Review(BaseModel):
    key_themes: List[str] = Field(description="Key themes discussed in the review")
    summary: str = Field(description="Brief summary")
    sentiment: Literal["pos", "neg"] = Field(description="Sentiment: pos or neg")
    pros: Optional[List[str]] = Field(default=None, description="List of pros")
    cons: Optional[List[str]] = Field(default=None, description="List of cons")
    name: Optional[str] = Field(default=None, description="Reviewer name")

# ✅ Parser
parser = PydanticOutputParser(pydantic_object=Review)

# ✅ Input
text = """I recently upgraded to the Samsung Galaxy S24 Ultra, and I must say, it’s an absolute powerhouse! The Snapdragon 8 Gen 3 processor makes everything lightning fast...

Review by Kunal 
"""

# ✅ Prompt (auto format instructions 🔥)
prompt = f"""
Extract structured information from the review.

{parser.get_format_instructions()}

Review:
{text}
"""

# ✅ Call LLM
response = model.invoke(prompt)

# ✅ Parse into Pydantic object
result: Review = parser.parse(response.content)

# ✅ Output
print(result.name)
print(result.summary)
print(result.sentiment)
print(result.pros)
print(result.cons)
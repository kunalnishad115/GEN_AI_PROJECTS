#  using the with_structured_output function to create a structured output for the LLM response
# import pydantic
# from pydantic import BaseModel, Field
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import JsonOutputParser
from dotenv import load_dotenv
from typing import TypedDict, Annotated, Optional, Literal
import os




load_dotenv()

# ✅ LLM (Groq config)
model = ChatOpenAI(
    model="llama-3.1-8b-instant",
    openai_api_key=os.getenv("GROQ_KEY"),
    openai_api_base="https://api.groq.com/openai/v1"
)

# ✅ Schema
class Review(TypedDict):
    key_themes: Annotated[list[str], "Key themes discussed in the review"]
    summary: Annotated[str, "Brief summary"]
    sentiment: Annotated[Literal["pos", "neg"], "Sentiment: pos or neg"]
    pros: Annotated[Optional[list[str]], "List of pros"]
    cons: Annotated[Optional[list[str]], "List of cons"]
    name: Annotated[Optional[str], "Reviewer name"]

# ✅ Parser
parser = JsonOutputParser()

# ✅ Input text
text = """I recently upgraded to the Samsung Galaxy S24 Ultra, and I must say, it’s an absolute powerhouse! The Snapdragon 8 Gen 3 processor makes everything lightning fast...

Review by Kunal 
"""

# ✅ Prompt (VERY IMPORTANT)
prompt = f"""
Extract structured information from the review.

Return ONLY valid JSON in this format:
{{
  "key_themes": [],
  "summary": "",
  "sentiment": "pos or neg",
  "pros": [],
  "cons": [],
  "name": ""
}}

Rules:
- sentiment must be only "pos" or "neg"
- Extract reviewer name if present
- Do not add extra text

Review:
{text}
"""

# ✅ Invoke
response = model.invoke(prompt)

# ✅ Parse
result: Review = parser.parse(response.content)

# ✅ Output
print(result["name"])
print(result["summary"])
print(result["sentiment"])
print(result["pros"])
print(result["cons"])
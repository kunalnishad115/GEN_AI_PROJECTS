from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser, PydanticOutputParser
from langchain_core.runnables import RunnableBranch, RunnableLambda
from pydantic import BaseModel, Field
from typing import Literal
import os

load_dotenv()

# ✅ Groq Model
model = ChatOpenAI(
    model="llama-3.1-8b-instant",
    openai_api_key=os.getenv("GROQ_KEY"),
    openai_api_base="https://api.groq.com/openai/v1"
)

parser = StrOutputParser()

# ✅ Schema
class Feedback(BaseModel):
    sentiment: Literal['positive', 'negative'] = Field(
        description='Sentiment of the feedback'
    )

parser2 = PydanticOutputParser(pydantic_object=Feedback)

# ✅ Classification Prompt
prompt1 = PromptTemplate(
    template="""
Classify the sentiment of the following feedback into positive or negative.

{format_instruction}

Feedback:
{feedback}
""",
    input_variables=['feedback'],
    partial_variables={
        'format_instruction': parser2.get_format_instructions()
    }
)

classifier_chain = prompt1 | model | parser2

# ✅ Response Prompts
prompt2 = PromptTemplate(
    template="Write an appropriate response to this positive feedback:\n{feedback}",
    input_variables=['feedback']
)

prompt3 = PromptTemplate(
    template="Write an appropriate response to this negative feedback:\n{feedback}",
    input_variables=['feedback']
)


branch_chain = RunnableBranch(
    (lambda x:x.sentiment == 'positive', prompt2 | model | parser),
    (lambda x:x.sentiment == 'negative', prompt3 | model | parser),
    RunnableLambda(lambda x: "could not find sentiment")
)

chain = classifier_chain | branch_chain

# ✅ Run
result = chain.invoke({
    "feedback": "This is a beautiful phone"
})

print(result)

# Optional: visualize graph
# chain.get_graph().print_ascii()
# from langchain_openai import ChatOpenAI   # removed bt langchain in new versions structured output parsers are in langchain_core
# from dotenv import load_dotenv
# from langchain_core.prompts import PromptTemplate
# # from langchain.output_parsers.structured import StructuredOutputParser, ResponseSchema
# from langchain.output_parsers.structured import StructuredOutputParser, ResponseSchema
# import os

# load_dotenv()

# # ✅ Groq Model
# model = ChatOpenAI(
#     model="llama-3.1-8b-instant",
#     openai_api_key=os.getenv("GROQ_KEY"),
#     openai_api_base="https://api.groq.com/openai/v1"
# )

# # ✅ Define Schema (Flexible 🔥)
# response_schemas = [
#     ResponseSchema(name="fact1", description="First fact about the topic"),
#     ResponseSchema(name="fact2", description="Second fact"),
#     ResponseSchema(name="fact3", description="Third fact"),
#     ResponseSchema(name="fact4", description="Fourth fact"),
#     ResponseSchema(name="fact5", description="Fifth fact"),
# ]

# # ✅ Structured Parser
# parser = StructuredOutputParser.from_response_schemas(response_schemas)

# # ✅ Format Instructions
# format_instructions = parser.get_format_instructions()

# # ✅ Prompt Template
# template = PromptTemplate(
#     template="""
# Give me 5 facts about {topic}.

# {format_instructions}
# """,
#     input_variables=["topic"],
#     partial_variables={"format_instructions": format_instructions}
# )

# # ✅ LCEL Chain
# chain = template | model | parser

# # ✅ Invoke
# result = chain.invoke({"topic": "black hole"})

# print(result)
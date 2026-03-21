from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from dotenv import load_dotenv
import os

load_dotenv()

llm = ChatOpenAI(
    model="llama-3.1-8b-instant",
    openai_api_key=os.getenv("GROQ_KEY"),
    openai_api_base="https://api.groq.com/openai/v1"
)

print("(type 'exit' to quit)\n")
chat_history = [
    SystemMessage(content="You are a helpful assistant."),
]
while True:
    user_input = input("You: ")
    chat_history.append(HumanMessage(content=user_input))

    if user_input.lower() == "exit":
        print("Goodbye 👋")
        break
    
    response = llm.invoke(chat_history)
    chat_history.append(AIMessage(content=response.content))

    print("AI:", response.content)

print("Chat history:", chat_history)

import streamlit as st
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
import os

# Load env
load_dotenv()

# Configure LLM (Groq)
llm = ChatOpenAI(
    model="llama-3.1-8b-instant",   # fast + free
    openai_api_key=os.getenv("GROQ_KEY"),
    openai_api_base="https://api.groq.com/openai/v1"
)

# UI
st.set_page_config(page_title="Research Paper Summarizer", layout="centered")

st.title("📄 Research Paper Summarizer (Groq LLM)")
st.write("Paste your research paper content below and get a summary.")

# Input text area
user_input = st.text_area(
    "Enter Research Paper Text",
    height=300,
    placeholder="Paste your research paper here..."
)

# Button
if st.button("Summarize"):
    if user_input.strip() == "":
        st.warning("Please enter some text first!")
    else:
        with st.spinner("Generating summary..."):
            prompt = f"Summarize the following research paper in simple terms:\n\n{user_input}"
            
            response = llm.invoke(prompt)
            
            st.subheader("📌 Summary:")
            st.write(response.content)
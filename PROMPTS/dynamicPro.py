import streamlit as st
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
import os

# Load env
load_dotenv()

# LLM (Groq)
llm = ChatOpenAI(
    model="llama-3.1-8b-instant",
    openai_api_key=os.getenv("GROQ_KEY"),
    openai_api_base="https://api.groq.com/openai/v1"
)

# UI Config
st.set_page_config(page_title="Dynamic Research Summarizer", layout="centered")
st.title("🧠 Dynamic Research Paper Summarizer")
st.write("Customize your summary using the options below 👇")

# 📌 1. Category Selector
category = st.selectbox(
    "Select Research Domain",
    ["Cancer Related", "Tech Related", "Chemistry", "Neuroscience"]
)

# 📌 2. Length Selector
length = st.selectbox(
    "Select Summary Length",
    ["Short", "Medium", "Long"]
)

# 📌 3. Style Selector
style = st.selectbox(
    "Select Writing Style",
    ["Simple", "Creative"]
)

# 📌 Input text
user_input = st.text_area(
    "Paste Research Paper",
    height=250,
    placeholder="Paste your research paper content here..."
)

# 📌 Button
if st.button("Generate Summary"):
    if user_input.strip() == "":
        st.warning("Please enter research paper content!")
    else:
        with st.spinner("Generating summary..."):

            # 🔥 Dynamic Prompt Engineering
            prompt = f"""
            You are an expert in {category} research.

            Summarize the following research paper.

            Requirements:
            - Length: {length}
            - Style: {style}
            - Make it clear and useful

            Research Paper:
            {user_input}
            """

            response = llm.invoke(prompt)

            st.subheader("📌 Generated Summary")
            st.write(response.content)
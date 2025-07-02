import streamlit as st
import os
from dotenv import load_dotenv
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.chains import RetrievalQA
from langchain_groq import ChatGroq
import tempfile

# Load environment variables
load_dotenv()
groq_api_key = os.getenv("GROQ_API_KEY")

st.set_page_config(page_title="📚 Chat with Timesheet", layout="wide")
st.title("🤖 Chatbot | Ask Questions about Timesheet")

# Check for uploaded file from main page
if "uploaded_file_bytes" not in st.session_state:
    st.warning("Please upload a timesheet file from the Upload & Analyze page first.")
    st.stop()

# Read text data
timesheet_text = st.session_state["uploaded_file_bytes"].decode("utf-8")

# Save temporary file for FAISS indexing
with tempfile.NamedTemporaryFile(delete=False, suffix=".txt") as tmp_file:
    tmp_file.write(timesheet_text.encode("utf-8"))
    temp_path = tmp_file.name

# Embed and create vector DB
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
vectorstore = FAISS.from_texts([timesheet_text], embedding=embeddings)
retriever = vectorstore.as_retriever()

# Setup QA system
llm = ChatGroq(temperature=0.3, model_name="llama3-8b-8192", groq_api_key=groq_api_key)
qa_chain = RetrievalQA.from_chain_type(llm=llm, retriever=retriever)

# Chat interface
user_question = st.text_input("💬 Ask a question about the timesheet report")
if user_question:
    with st.spinner("Thinking..."):
        response = qa_chain.run(user_question)
    st.markdown("### 🤖 Response")
    st.write(response)

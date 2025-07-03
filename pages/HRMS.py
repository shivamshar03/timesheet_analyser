import streamlit as st
from pymongo import MongoClient
from sentence_transformers import SentenceTransformer
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain_groq import ChatGroq
import os
from dotenv import load_dotenv

# ---------- CONFIG ----------
load_dotenv()

# ---------- STREAMLIT UI ----------
st.set_page_config(page_title="HRMS chat", layout="wide")
st.title("Chat With HRMS")
st.markdown("Ask any question about the `HRMS.users` collection.")

# ---------- MONGODB ----------
@st.cache_resource
def load_documents():
    uri = "mongodb+srv://naveencarinasoftlabs:root@cluster0.vxfp99z.mongodb.net"
    client = MongoClient(uri)
    collection = client["HRMS"]["users"]
    docs = [str(doc) for doc in collection.find({})]
    return docs

with st.spinner("🔄 Loading data from MongoDB..."):
    documents = load_documents()

# ---------- EMBEDDINGS & FAISS ----------
@st.cache_resource
def build_vectorstore(docs):
    embedding_fn = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    vectorstore = FAISS.from_texts(docs, embedding_fn)
    return vectorstore

with st.spinner("📊 Creating embeddings and indexing..."):
    vectorstore = build_vectorstore(documents)

# ---------- RETRIEVAL QA ----------
@st.cache_resource
def load_qa_chain():
    retriever = vectorstore.as_retriever()
    qa = RetrievalQA.from_chain_type(
        llm=ChatGroq(temperature=0.3, model_name="llama3-8b-8192"),  # Uses your OPENAI_API_KEY from .env or env vars
        retriever=retriever,
        chain_type="stuff"
    )
    return qa

qa_chain = load_qa_chain()

# ---------- USER INPUT ----------
query = st.text_input("🔎 Your Question:")

if query:
    with st.spinner("🧠 Thinking..."):
        response = qa_chain.run(query)
        st.success("💬 Answer:")
        st.write(response)

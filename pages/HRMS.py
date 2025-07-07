import os
import shutil
import streamlit as st
from pymongo import MongoClient
from dotenv import load_dotenv
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain_groq import ChatGroq
from langchain.schema import Document
import pandas as pd

# ------------------- Load Env -------------------
load_dotenv()
MONGO_URI = "mongodb+srv://naveencarinasoftlabs:root@cluster0.vxfp99z.mongodb.net"

# ------------------- Streamlit UI -------------------
st.set_page_config(page_title="HRMS Assistant", layout="wide")
tabs = st.tabs(["💬 Chat", "📄 View All Users", "🛠️ Rebuild Index"])

# ------------------- MongoDB -------------------
client = MongoClient(MONGO_URI)
collection = client["HRMS"]["users"]

# ------------------- Format MongoDB Docs -------------------
def load_documents():
    raw_docs = collection.find()
    documents = []
    for doc in raw_docs:
        doc_str = (
            f"Name: {doc.get('name', 'N/A')}\n"
            f"Email: {doc.get('email', 'N/A')}\n"
            f"Role: {doc.get('role', 'N/A')}\n"
            f"Location: {doc.get('location', 'N/A')}\n"
            f"Join Date: {doc.get('joined_date', 'N/A')}"
        )
        documents.append(doc_str)
    return documents

# ------------------- Build or Load FAISS -------------------
def build_vectorstore(docs):
    index_path = "faiss_index"
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    docs = [Document(page_content=doc) for doc in docs]
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = splitter.split_documents(docs)
    vectorstore = FAISS.from_documents(chunks, embeddings)
    vectorstore.save_local(index_path)
    return vectorstore

def get_vectorstore(force_rebuild=False):
    index_path = "faiss_index"
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

    if force_rebuild and os.path.exists(index_path):
        shutil.rmtree(index_path)

    if os.path.exists(index_path):
        try:
            return FAISS.load_local(index_path, embeddings, allow_dangerous_deserialization=True)
        except:
            st.warning("⚠️ Failed to load index. Rebuilding...")

    docs = load_documents()
    return build_vectorstore(docs)

# ------------------- Conversation Chain -------------------
def get_conversational_chain():
    vectorstore = get_vectorstore()
    retriever = vectorstore.as_retriever(search_kwargs={"k": 100})
    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
    llm = ChatGroq(model_name="llama3-8b-8192", temperature=0)
    return ConversationalRetrievalChain.from_llm(llm, retriever, memory=memory)

# ------------------- 💬 Chat Tab -------------------
with tabs[0]:
    st.header("💬 Chat with HRMS (Memory + Pagination)")

    if "chat_chain" not in st.session_state:
        st.session_state.chat_chain = get_conversational_chain()
        st.session_state.chat_history = []
        st.session_state.chat_pages = []

    query = st.chat_input("Ask a question about HRMS.users")

    if query:
        with st.spinner("Thinking..."):
            response = st.session_state.chat_chain.run(query)
            st.write(response)

# ------------------- 📄 View All Users Tab -------------------
with tabs[1]:
    st.header("📄 All Users in HRMS.users")

    data = list(collection.find())
    if data:
        df = pd.DataFrame(data)
        df.drop(columns=["_id"], inplace=True, errors="ignore")
        st.dataframe(df, use_container_width=True)
        st.info(f"✅ Total Records: {len(df)}")

        usernames = df["name"].dropna().unique().tolist()
        selected_name = st.selectbox("👤 View profile for:", ["-- Select User --"] + usernames)

        if selected_name != "-- Select User --":
            selected_user = df[df["name"] == selected_name].iloc[0]
            st.subheader(f"🧑‍💼 Profile of {selected_name}")
            for col in selected_user.index:
                st.markdown(f"**{col}:** {selected_user[col]}")
    else:
        st.warning("⚠️ No data found in collection.")

# ------------------- 🛠️ Rebuild Vector DB Tab -------------------
with tabs[2]:
    st.header("🛠️ Rebuild FAISS Vector Index")
    if st.button("🔄 Clear & Rebuild Index"):
        get_vectorstore(force_rebuild=True)
        st.success("✅ Rebuilt vector DB using all documents.")

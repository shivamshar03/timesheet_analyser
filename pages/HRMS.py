import os
import shutil
import streamlit as st
from pymongo import MongoClient
from dotenv import load_dotenv
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain.schema import Document
from langchain_groq import ChatGroq
import pandas as pd
from bson import ObjectId

# ------------------- Load Env -------------------
load_dotenv()
MONGO_URI = os.getenv("MONGO_URI", "mongodb+srv://naveencarinasoftlabs:root@cluster0.vxfp99z.mongodb.net")

# ------------------- MongoDB -------------------
client = MongoClient(MONGO_URI)
db = client["HRMS"]
all_collections = [col for col in db.list_collection_names() if not col.startswith("system.")]

# ------------------- Streamlit UI -------------------
st.set_page_config(page_title="HRMS Assistant", layout="wide")
st.sidebar.title("🗂️ Collection Settings")
selected_collection_name = st.sidebar.selectbox("📂 Select Collection", all_collections)
collection = db[selected_collection_name]

tabs = st.tabs(["💬 Chat", "📄 View Records", "🛠️ Rebuild Index"])

# ------------------- MongoDB Data Sanitizer -------------------
def sanitize_mongo_data(data):
    def sanitize_value(val):
        if isinstance(val, ObjectId):
            return str(val)
        elif isinstance(val, dict):
            return {k: sanitize_value(v) for k, v in val.items()}
        elif isinstance(val, list):
            return [sanitize_value(v) for v in val]
        else:
            return val

    return [{k: sanitize_value(v) for k, v in doc.items()} for doc in data]

# ------------------- Load MongoDB Documents -------------------
def load_documents():
    raw_docs = collection.find()
    documents = []

    for doc in raw_docs:
        try:
            doc.pop("_id", None)
            doc = sanitize_mongo_data([doc])[0]  # Sanitize each doc
            doc_str = "\n".join([f"{k}: {v}" for k, v in doc.items()])
            documents.append(doc_str)
        except Exception as e:
            st.warning(f"❌ Skipping document due to error: {e}")
    return documents

# ------------------- FAISS Index Functions -------------------
def build_vectorstore(docs, collection_name):
    index_path = f"faiss_index/{collection_name}"
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    docs = [Document(page_content=doc) for doc in docs]
    splitter = RecursiveCharacterTextSplitter(chunk_size=400, chunk_overlap=50)
    chunks = splitter.split_documents(docs)
    st.info(f"🧩 Total chunks created: {len(chunks)}")
    vectorstore = FAISS.from_documents(chunks, embeddings)
    vectorstore.save_local(index_path)
    return vectorstore

def get_vectorstore(force_rebuild=False):
    index_path = f"faiss_index/{selected_collection_name}"
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

    if force_rebuild and os.path.exists(index_path):
        shutil.rmtree(index_path)

    if os.path.exists(index_path):
        try:
            return FAISS.load_local(index_path, embeddings, allow_dangerous_deserialization=True)
        except Exception as e:
            st.warning(f"⚠️ Failed to load index: {e}. Rebuilding...")

    docs = load_documents()
    if not docs:
        st.error("❌ No documents found to index.")
        return None
    return build_vectorstore(docs, selected_collection_name)

# ------------------- LLM Chain -------------------
def get_llm_chain():
    llm = ChatGroq(model_name="llama3-8b-8192", temperature=0)
    prompt_template = PromptTemplate(
        input_variables=["context", "question"],
        template="""
You are an assistant for the HRMS database. Use the following context to answer the user's question.
If the answer is not found in the context, reply "I couldn't find that information."

Context:
{context}

Question: {question}
"""
    )
    return LLMChain(llm=llm, prompt=prompt_template)

# ------------------- 💬 Chat Tab -------------------
with tabs[0]:
    st.header(f"💬 Chat with HRMS.{selected_collection_name}")

    if "vectorstore" not in st.session_state or st.session_state.get("collection_name") != selected_collection_name:
        with st.spinner("🔍 Loading vector store..."):
            st.session_state.vectorstore = get_vectorstore()
            st.session_state.collection_name = selected_collection_name

    if "chat_chain" not in st.session_state:
        st.session_state.chat_chain = get_llm_chain()

    query = st.chat_input("Ask something about the selected collection...")

    if query:
        with st.spinner("🤖 Thinking..."):
            docs = st.session_state.vectorstore.similarity_search(query, k=4)
            context = "\n\n".join([doc.page_content for doc in docs])
            response = st.session_state.chat_chain.run({
                "context": context,
                "question": query
            })
            st.markdown("**🧠 Answer:**")
            st.write(response)

            with st.expander("📄 Retrieved context"):
                st.write(context)

# ------------------- 📄 View Records Tab -------------------
with tabs[1]:
    st.header(f"📄 All Records in HRMS.{selected_collection_name}")
    data = list(collection.find())
    if data:
        df = pd.DataFrame(sanitize_mongo_data(data))
        st.dataframe(df, use_container_width=True)
        st.info(f"✅ Total Records: {len(df)}")

        if "name" in df.columns:
            usernames = df["name"].dropna().unique().tolist()
            selected_name = st.selectbox("👤 View profile for:", ["-- Select --"] + usernames)
            if selected_name != "-- Select --":
                selected_user = df[df["name"] == selected_name].iloc[0]
                st.subheader(f"🧑‍💼 Profile of {selected_name}")
                for col in selected_user.index:
                    st.markdown(f"**{col}:** {selected_user[col]}")
    else:
        st.warning("⚠️ No data found in collection.")

# ------------------- 🛠️ Rebuild Vector Index Tab -------------------
with tabs[2]:
    st.header(f"🛠️ Rebuild FAISS Index for HRMS.{selected_collection_name}")
    if st.button("🔄 Clear & Rebuild Index"):
        st.session_state.vectorstore = get_vectorstore(force_rebuild=True)
        st.session_state.collection_name = selected_collection_name
        st.success(f"✅ Vector DB rebuilt for {selected_collection_name}.")

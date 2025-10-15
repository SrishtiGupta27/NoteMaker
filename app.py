import os
import streamlit as st
from dotenv import load_dotenv
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
from langchain_google_community import GoogleDriveLoader
from langchain_groq import ChatGroq
from googleapiclient.discovery import build
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow


# CONFIG

load_dotenv()

TRANSCRIPTS_DIR = "data/transcripts"
VECTORS_DIR = "data/vectors"
os.makedirs(TRANSCRIPTS_DIR, exist_ok=True)
os.makedirs(VECTORS_DIR, exist_ok=True)

# Hard-coded Google Drive folder ID
FOLDER_ID = "1Uzzwkc1eYOOrSsBxkd8BcVRUgPebTYVx"

# LLM and embeddings
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
if not GROQ_API_KEY:
    raise ValueError("GROQ_API_KEY not found in .env")

embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
llm = ChatGroq(model_name="meta-llama/llama-4-maverick-17b-128e-instruct", groq_api_key=GROQ_API_KEY)
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)

# Google Drive OAuth
SCOPES = ["https://www.googleapis.com/auth/drive"]
CREDS_PATH = ".credentials/credentials.json"
TOKEN_PATH = ".credentials/token.json"

def authorize_google_drive():
    os.makedirs(".credentials", exist_ok=True)
    if not os.path.exists(TOKEN_PATH):
        flow = InstalledAppFlow.from_client_secrets_file(CREDS_PATH, SCOPES)
        creds = flow.run_local_server(port=0)
        with open(TOKEN_PATH, "w") as token_file:
            token_file.write(creds.to_json())

authorize_google_drive()


# FUNCTIONS


def fetch_new_google_docs():
    loader = GoogleDriveLoader(
        folder_id=FOLDER_ID,
        recursive=False,
        credentials_path=CREDS_PATH,
        token_path=TOKEN_PATH,
        load_auth=True,
        file_types=["document", "pdf", "presentation"],
    )

    docs = loader.load()
    new_files_count = 0
    new_files_list = []

    for i, doc in enumerate(docs):
        file_id = doc.metadata.get("id", f"id_{i+1}")
        safe_title = doc.metadata.get("name", f"doc_{i+1}").replace("/", "_").replace(" ", "_")
        filename = f"{file_id}_{safe_title}.txt"
        filepath = os.path.join(TRANSCRIPTS_DIR, filename)

        if not os.path.exists(filepath):
            with open(filepath, "w", encoding="utf-8") as f:
                f.write(doc.page_content)
            new_files_count += 1
            new_files_list.append(filename)

    return new_files_count, new_files_list


def build_or_load_vectors():
    vector_stores = []
    for file in os.listdir(TRANSCRIPTS_DIR):
        if not file.endswith(".txt"):
            continue

        base_name = os.path.splitext(file)[0]
        faiss_path = os.path.join(VECTORS_DIR, f"{base_name}.faiss")
        pkl_path = os.path.join(VECTORS_DIR, f"{base_name}.pkl")

        if os.path.exists(faiss_path) and os.path.exists(pkl_path):
            db = FAISS.load_local(VECTORS_DIR, embeddings, index_name=base_name, allow_dangerous_deserialization=True)
        else:
            with open(os.path.join(TRANSCRIPTS_DIR, file), "r", encoding="utf-8") as f:
                text = f.read()
            chunks = text_splitter.split_text(text)
            db = FAISS.from_texts(chunks, embeddings)
            db.save_local(VECTORS_DIR, index_name=base_name)

        vector_stores.append(db)

    if not vector_stores:
        st.error("No vector stores found. Exiting.")
        return None

    combined_db = vector_stores[0]
    for db in vector_stores[1:]:
        combined_db.merge_from(db)
    return combined_db


def get_answer_from_db(db, query):
    retriever = db.as_retriever(search_kwargs={"k": 5})
    docs = retriever.get_relevant_documents(query)
    context = "\n".join([d.page_content for d in docs])

    prompt = f"""
<TASK>
You are an AI assistant. Answer the user's question using ONLY the knowledge extracted from the provided context (vectorized transcripts).
Focus on providing accurate, concise, and professional answers.
</TASK>

<RESTRICTIONS>
- Do not reference any names, personal details, or sensitive information from the transcripts
- Do not make assumptions beyond the context provided
- Answer strictly based on the knowledge base
</RESTRICTIONS>

<CONTEXT>
{context}
</CONTEXT>

<QUESTION>
{query}
</QUESTION>

<INSTRUCTIONS>
- Use clear and professional language
- Keep answers concise, factual, and neutral
- Never reveal names or personal identifiers
</INSTRUCTIONS>

<ANSWER>
"""
    response = llm.invoke(prompt)
    return response.content

# STREAMLIT APP


st.set_page_config(page_title="Transcript QA", layout="wide")
st.title(" Google Docs Transcript Q&A")

# Step 1: Fetch new documents
st.header("Step 1: Fetch new Google Docs")
if st.button("Fetch Documents from Drive"):
    new_count, new_files = fetch_new_google_docs()
    st.success(f"Fetched {new_count} new transcripts.")
    if new_files:
        st.write("New files added:", new_files)

# Step 2: Build/load vectors
st.header("Step 2: Build/load vector knowledge base")
if st.button("Build/Load Vectors"):
    db = build_or_load_vectors()
    if db:
        st.success("Knowledge base ready for Q&A!")
    else:
        st.error("Failed to build vector knowledge base.")

# Step 3: Ask questions
st.header("Step 3: Ask Questions")
if "db" not in st.session_state:
    st.session_state.db = build_or_load_vectors()

query = st.text_input("Ask a question across all transcripts:")
if query and st.session_state.db:
    with st.spinner("Fetching answer..."):
        answer = get_answer_from_db(st.session_state.db, query)
    st.markdown(f"**Answer:** {answer}")

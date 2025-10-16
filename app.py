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
You are an AI meeting assistant called "NoteMaker".
Respond in the **first person** and maintain a natural, conversational tone.
Make it clear that you are an AI assistant, but speak directly to the user — not as an observer of others.
Your responses must be based ONLY on the information provided in the context (vectorized transcripts).
</TASK>

<STRICT PRIVACY & CONTENT RULES>
NEVER mention, quote, or refer to:
- Any names, people, or participants
- Phrases like “someone said”, “a team member mentioned”, or similar
- Any private, sensitive, or identifying details
- Any company names, internal project titles, or specific data

ALWAYS:
- Speak as if you are directly assisting the user.
- Offer clear, factual, and supportive guidance or answers.
- Use first-person tone (“I recommend…”, “I suggest…”, “Here’s what you can do…”).
- Keep the response concise and professional.
- Stay within the information in the context — no assumptions or inventions.

If the context does not provide enough information to answer, say:
“I don’t have enough details to answer that precisely, but I can suggest some general steps if you’d like.”
</STRICT PRIVACY & CONTENT RULES>

<CONTEXT>
{context}
</CONTEXT>

<QUESTION>
{query}
</QUESTION>

<INSTRUCTIONS>
- Begin by acknowledging you are an AI assistant (“I’m an AI assistant here to help…”).
- Speak **directly** to the user with advice, explanation, or insight.
- Do **not** mention transcripts, participants, or discussions.
- Focus on providing value, not meta-description.
</INSTRUCTIONS>

<ANSWER>
"""





    response = llm.invoke(prompt)
    return response.content



# STREAMLIT APP (ENHANCED UI)


st.set_page_config(page_title="NoteMaker", layout="wide", page_icon="")

st.markdown(
    """
    <h1 style='text-align: center; color: #2C3E50;'> NoteMaker — Meeting Summarizer & Q&A Assistant</h1>
    <p style='text-align: center; color: gray;'>Fetch meeting notes from Google Drive and ask intelligent questions from your transcripts.</p>
    <hr>
    """,
    unsafe_allow_html=True,
)

# Initialize session state
if "db" not in st.session_state:
    st.session_state.db = build_or_load_vectors()
if "messages" not in st.session_state:
    st.session_state.messages = []

# Sidebar for actions
with st.sidebar:
    st.header(" Document Management")
    if st.button(" Fetch Documents"):
        with st.spinner("Fetching Google Docs..."):
            new_count, new_files = fetch_new_google_docs()
        st.success(f"Fetched {new_count} new transcript(s).")
        if new_files:
            st.write("New Files:", new_files)

    if st.button(" Build / Reload Vectors"):
        with st.spinner("Building FAISS vector store..."):
            st.session_state.db = build_or_load_vectors()
        if st.session_state.db:
            st.success("Knowledge base ready!")
        else:
            st.error("Failed to build vector knowledge base.")

# Main chat area
st.subheader(" Ask Questions")

query = st.chat_input("Type your question here...")

if query:
    st.session_state.messages.append({"role": "user", "content": query})
    if st.session_state.db:
        with st.spinner("Thinking..."):
            answer = get_answer_from_db(st.session_state.db, query)
        st.session_state.messages.append({"role": "assistant", "content": answer})
    else:
        st.warning(" Please build or load the knowledge base first.")

# Display chat conversation
for msg in st.session_state.messages:
    if msg["role"] == "user":
        with st.chat_message("user"):
            st.markdown(msg["content"])
    else:
        with st.chat_message("assistant"):
            st.markdown(msg["content"])

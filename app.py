import streamlit as st
import os
import io
from dotenv import load_dotenv
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_groq import ChatGroq
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseDownload

# CONFIGURATION

load_dotenv()

BASE_DIR = "data"
SUMMARY_DIR = os.path.join(BASE_DIR, "summary")  
VECTOR_DIR = os.path.join(BASE_DIR, "vectors")
os.makedirs(SUMMARY_DIR, exist_ok=True)
os.makedirs(VECTOR_DIR, exist_ok=True)

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
if not GROQ_API_KEY:
    raise ValueError("GROQ_API_KEY not found in .env file")

# Google Drive parent folder ID (user_summary)
USER_SUMMARY_PARENT_FOLDER_ID = os.getenv("USER_SUMMARY_PARENT_FOLDER_ID")
if not USER_SUMMARY_PARENT_FOLDER_ID:
    raise ValueError("USER_SUMMARY_PARENT_FOLDER_ID not found in .env file")

# LLM and embeddings
llm = ChatGroq(model_name="meta-llama/llama-4-maverick-17b-128e-instruct",
               groq_api_key=GROQ_API_KEY)
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)

# OAuth files
CREDS_PATH = ".credentials/credentials.json"
TOKEN_PATH = ".credentials/token.json"
SCOPES = ["https://www.googleapis.com/auth/drive"]


# GOOGLE DRIVE AUTH

def authorize_google_drive():
    """Authenticate and return Google Drive service"""
    os.makedirs(".credentials", exist_ok=True)

    creds = None
    if os.path.exists(TOKEN_PATH):
        creds = Credentials.from_authorized_user_file(TOKEN_PATH, SCOPES)
    if not creds or not creds.valid:
        flow = InstalledAppFlow.from_client_secrets_file(CREDS_PATH, SCOPES)
        creds = flow.run_local_server(port=0)
        with open(TOKEN_PATH, "w") as token_file:
            token_file.write(creds.to_json())
    service = build("drive", "v3", credentials=creds)
    return service


# HELPER FUNCTIONS

def get_user_paths(user_id):
    user_summary_dir = os.path.join(SUMMARY_DIR, user_id)
    user_vector_dir = os.path.join(VECTOR_DIR, user_id)
    os.makedirs(user_summary_dir, exist_ok=True)
    os.makedirs(user_vector_dir, exist_ok=True)
    return user_summary_dir, user_vector_dir


def fetch_user_txt_summaries(service, user_id):
    """
    Fetch all .txt files from the user_id folder under USER_SUMMARY_PARENT_FOLDER_ID
    """
    print(f"Fetching summaries for user: {user_id} from Drive...")

    query = f"'{USER_SUMMARY_PARENT_FOLDER_ID}' in parents and name='{user_id}' and mimeType='application/vnd.google-apps.folder'"
    results = service.files().list(q=query, fields="files(id, name)").execute()
    folders = results.get("files", [])
    if not folders:
        print(f"No folder found for user {user_id} in parent folder.")
        return []
    user_folder_id = folders[0]["id"]

    query = f"'{user_folder_id}' in parents and mimeType='text/plain'"
    results = service.files().list(q=query, fields="files(id, name)").execute()
    files = results.get("files", [])
    if not files:
        print(f"No .txt files found in user folder {user_id}")
        return []

    user_summary_dir, _ = get_user_paths(user_id)
    local_paths = []
    for f in files:
        file_id = f["id"]
        file_name = f["name"]
        request = service.files().get_media(fileId=file_id)
        fh = io.FileIO(os.path.join(user_summary_dir, file_name), "wb")
        downloader = MediaIoBaseDownload(fh, request)
        done = False
        while not done:
            status, done = downloader.next_chunk()
        fh.close()
        local_paths.append(os.path.join(user_summary_dir, file_name))

    print(f"Fetched {len(local_paths)} summaries for {user_id}.")
    return local_paths


def generate_cumulative_summary(user_id):
    user_summary_dir, _ = get_user_paths(user_id)
    cum_summary_path = os.path.join(user_summary_dir, "add_summary.txt")

    session_files = [
        os.path.join(user_summary_dir, f)
        for f in os.listdir(user_summary_dir)
        if f.endswith(".txt") and f != "add_summary.txt"
    ]
    if not session_files:
        print(f"No new session summaries for user {user_id}")
        return None

    old_cum = ""
    if os.path.exists(cum_summary_path):
        with open(cum_summary_path, "r", encoding="utf-8") as f:
            old_cum = f.read()

    new_summaries_text = "\n\n".join(open(f, "r", encoding="utf-8").read() for f in session_files)

    prompt = f"""
You are an AI assistant tasked with maintaining a **cumulative summary** for a single user.
You have the **previous cumulative summary** and the **new session summaries**.

<OLD_SUMMARY>
{old_cum}
</OLD_SUMMARY>

<NEW_SUMMARIES>
{new_summaries_text}
</NEW_SUMMARIES>

Instructions:
1. Combine the old summary and new summaries into a single cumulative summary.
2. **Preserve all key points, decisions, action items, and important insights** from both old and new.
3. Do not remove, skip, or invent any details unless clearly redundant.
4. Organize the summary clearly (you can use bullet points or sections for readability).
5. Keep the summary concise but complete — focus on retaining all essential information.
6. Maintain a neutral and factual tone suitable for later reference.

Output the refined cumulative summary. Do not include any text outside the summary.
"""
    print(f"Generating cumulative summary for {user_id} ...")
    response = llm.invoke(prompt)
    new_cum_summary = response.content.strip()

    with open(cum_summary_path, "w", encoding="utf-8") as f:
        f.write(new_cum_summary)

    for f in session_files:
        os.remove(f)

    print(f"Cumulative summary updated: {cum_summary_path}")
    return cum_summary_path


def vectorize_cumulative_summary(user_id):
    user_summary_dir, user_vector_dir = get_user_paths(user_id)
    cum_summary_path = os.path.join(user_summary_dir, "add_summary.txt")

    if not os.path.exists(cum_summary_path):
        print(f"No cumulative summary for {user_id}")
        return None

    with open(cum_summary_path, "r", encoding="utf-8") as f:
        text = f.read()

    chunks = text_splitter.split_text(text)
    db = FAISS.from_texts(chunks, embeddings)
    db.save_local(user_vector_dir, index_name="add_summary")
    print(f"Vector store saved for {user_id} → {user_vector_dir}")
    return db


def load_user_vector_db(user_id):
    _, user_vector_dir = get_user_paths(user_id)
    faiss_path = os.path.join(user_vector_dir, "add_summary.faiss")
    pkl_path = os.path.join(user_vector_dir, "add_summary.pkl")

    if not os.path.exists(faiss_path) or not os.path.exists(pkl_path):
        print(f"No vector DB found for {user_id}.")
        return None

    db = FAISS.load_local(user_vector_dir, embeddings, index_name="add_summary",
                          allow_dangerous_deserialization=True)
    return db


def get_answer_from_user_db(user_id, query):
    db = load_user_vector_db(user_id)
    if not db:
        return "No knowledge base found. Please fetch summaries and vectorize first."

    retriever = db.as_retriever(search_kwargs={"k": 5})
    docs = retriever.get_relevant_documents(query)
    context = "\n".join([d.page_content for d in docs])

    prompt = f"""
<TASK>
You are an AI meeting assistant called "NoteMaker". You are helping a single user (user ID: {user_id}) based on their cumulative session summaries.
Respond in the **first person** and maintain a natural, conversational tone.
Use the context to provide guidance, advice, or strategies.
You may give **general, safe instructions** such as lifestyle changes, behavioral strategies, stress management, or routines.
</TASK>

<STRICT MEDICATION RULES>
- NEVER suggest, recommend, or prescribe any medication on your own.
- Only mention medications if they are **explicitly mentioned in the provided context**.
- Under no circumstances invent or assume medication information.
</STRICT MEDICATION RULES>

<STRICT PRIVACY & CONTENT RULES>
NEVER mention, quote, or refer to:
- Any names, people, or participants
- Any private, sensitive, or identifying details
- Company names, internal project titles, or specific data
- Phrases like “someone said” or “a team member mentioned”

ALWAYS:
- Speak directly to the user in a supportive, first-person manner.
- Offer practical guidance, insights, or suggestions derived from the context and general safe advice.
- Keep responses concise, clear, and professional.
- Stay strictly within the information in the provided context for medications; invent nothing.
</STRICT PRIVACY & CONTENT RULES>

<CONTEXT>
{context}
</CONTEXT>

<QUESTION>
{query}
</QUESTION>

<INSTRUCTIONS>
- Begin by acknowledging you are an AI assistant (“I’m an AI assistant here to help…”).
- Give advice **based on context for medications only**; do not invent any medication instructions.
- You may provide **general, safe, non-medication guidance on your own**.
- Keep the tone supportive, clear, and conversational.
- Avoid repeating the question or including extra commentary; focus on guidance.
</INSTRUCTIONS>

<ANSWER>


"""
    response = llm.invoke(prompt)
    return response.content.strip()

# STREAMLIT APP 

import streamlit as st

st.set_page_config(page_title="NoteMaker AI Assistant", layout="wide")

# Sidebar: Initial Setup
with st.sidebar:
    st.header("⚙️ Setup & Data Sync")

    st.markdown("Configure your connection and sync summaries.")

    user_id = st.text_input("🧑 User ID", placeholder="Enter your user ID")

    st.divider()

    if st.button("🔐 Authorize Google Drive"):
        with st.spinner("Authorizing Google Drive..."):
            try:
                service = authorize_google_drive()
                st.session_state["service"] = service
                st.success("✅ Google Drive authorized successfully.")
            except Exception as e:
                st.error(f"Authorization failed: {e}")

    if "service" in st.session_state and user_id:
        if st.button("📥 Fetch & Update Summaries"):
            with st.spinner("Fetching and updating summaries..."):
                try:
                    fetch_user_txt_summaries(st.session_state["service"], user_id)
                    generate_cumulative_summary(user_id)
                    vectorize_cumulative_summary(user_id)
                    st.success("✅ Summaries updated and vectorized successfully.")
                except Exception as e:
                    st.error(f"❌ Failed: {e}")
    else:
        st.info("Enter your User ID and authorize Google Drive first.")

    st.divider()
    st.caption("💡 Tip: Once setup is complete, ask questions in the chat on the right.")

# Main Chat UI
st.title("💬 NoteMaker AI Assistant")
st.caption("An AI assistant that summarizes and answers based on your session history.")

# Initialize chat messages
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display existing chat messages
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# Chat input area
if prompt := st.chat_input("Ask something about your sessions..."):
    if not user_id:
        st.warning("⚠️ Please enter your User ID in the sidebar first.")
    else:
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                try:
                    answer = get_answer_from_user_db(user_id, prompt)
                    st.markdown(answer)
                    st.session_state.messages.append({"role": "assistant", "content": answer})
                except Exception as e:
                    st.error(f"Error: {e}")


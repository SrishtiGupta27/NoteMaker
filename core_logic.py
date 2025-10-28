import os
import io
import json
import time
from dotenv import load_dotenv
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_groq import ChatGroq
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseDownload
from pydantic import BaseModel
from typing import Dict, Any

# --- CONFIGURATION ---
load_dotenv()

BASE_DIR = "data"
TRANSCRIPT_DIR = os.path.join(BASE_DIR, "transcript") 
SUMMARY_DIR = os.path.join(BASE_DIR, "summary")  
VECTOR_DIR = os.path.join(BASE_DIR, "vectors")

os.makedirs(TRANSCRIPT_DIR, exist_ok=True) 
os.makedirs(SUMMARY_DIR, exist_ok=True)
os.makedirs(VECTOR_DIR, exist_ok=True)

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
if not GROQ_API_KEY:
    raise ValueError("GROQ_API_KEY not found in .env file")

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

# Mime Types to fetch/export
DOC_MIME_TYPES = {
    'application/vnd.google-apps.document': 'text/plain', # Google Doc -> TXT
    'application/vnd.google-apps.spreadsheet': 'text/csv', # Google Sheet -> CSV (simplification)
    'application/vnd.google-apps.file': 'application/vnd.google-apps.document', # Placeholder for general files
}
QUERY_MIME_FILTER = " or ".join([f"mimeType='{m}'" for m in DOC_MIME_TYPES.keys()]) + " or mimeType='text/plain'"


# --- GOOGLE DRIVE AUTH FOR FASTAPI ---

def get_auth_flow(request_url: str):
    """Sets up the flow and returns the authorization URL and state."""
    os.makedirs(".credentials", exist_ok=True)
    # The redirect_uri must match your FastAPI callback endpoint URL
    flow = InstalledAppFlow.from_client_secrets_file(
        CREDS_PATH, 
        scopes=SCOPES, 
        redirect_uri=request_url
    )
    return flow

def complete_authorization(flow, code: str):
    """Completes the authorization process using the received code."""
    flow.fetch_token(code=code)
    creds = flow.credentials
    with open(TOKEN_PATH, "w") as token_file:
        token_file.write(creds.to_json())
    service = build("drive", "v3", credentials=creds)
    return service

def get_drive_service_from_token():
    """Builds and returns Google Drive service from existing token."""
    if not os.path.exists(TOKEN_PATH):
        return None
    
    with open(TOKEN_PATH, "r") as token_file:
        creds_info = json.load(token_file)
    
    creds = Credentials.from_authorized_user_info(creds_info, SCOPES)
    if creds.valid:
        service = build("drive", "v3", credentials=creds)
        return service
    else:
        # NOTE: A real app should handle token refresh here
        return None 


# --- HELPER FUNCTIONS ---

def get_user_paths(user_id):
    user_transcript_dir = os.path.join(TRANSCRIPT_DIR, user_id) 
    user_summary_dir = os.path.join(SUMMARY_DIR, user_id)
    user_vector_dir = os.path.join(VECTOR_DIR, user_id)
    
    os.makedirs(user_transcript_dir, exist_ok=True)
    os.makedirs(user_summary_dir, exist_ok=True)
    os.makedirs(user_vector_dir, exist_ok=True)
    
    return user_transcript_dir, user_summary_dir, user_vector_dir


def generate_session_summary(raw_text, original_filename, user_id):
    """Generates a summary for a single document and saves it as a .txt file."""
    _, user_summary_dir, _ = get_user_paths(user_id) 
    
    session_summary_filename = f"{os.path.splitext(original_filename)[0]}_{int(time.time())}.txt"
    session_summary_path = os.path.join(user_summary_dir, session_summary_filename)

    # Use the full raw text (the file conversion should have already handled truncation/limits if necessary)
    prompt = f"""
You are an AI assistant tasked with creating a **concise session summary** from a raw document or meeting transcript.

<DOCUMENT_TEXT>
{raw_text} 
</DOCUMENT_TEXT>

Instructions:
1. Identify and extract **all key points, decisions, action items, and important insights**.
2. Organize the summary clearly using bullet points or sections.
3. Maintain a neutral and factual tone.
4. The summary will be used later to build a cumulative summary and answer user queries.

Output the concise session summary. Do not include any text outside the summary.
"""
    print(f"Generating session summary for: {original_filename} ...")
    response = llm.invoke(prompt)
    session_summary = response.content.strip()

    with open(session_summary_path, "w", encoding="utf-8") as f:
        f.write(session_summary)

    print(f"Session summary saved: {session_summary_path}")
    return session_summary_path


def fetch_and_process_session_files(service, user_id):
    """
    Fetch various document files, convert them to text, generate session summaries,
    and clean up the intermediate transcript files.
    """
    print(f"Fetching files for user: {user_id} from Drive...")
    
    # 1. Find the user's folder ID
    query = f"'{USER_SUMMARY_PARENT_FOLDER_ID}' in parents and name='{user_id}' and mimeType='application/vnd.google-apps.folder'"
    results = service.files().list(q=query, fields="files(id, name)").execute()
    folders = results.get("files", [])
    if not folders:
        print(f"No folder found for user {user_id} in parent folder.")
        return []
    user_folder_id = folders[0]["id"]

    # 2. Query for document files within the user's folder
    query = f"'{user_folder_id}' in parents and ({QUERY_MIME_FILTER})"
    results = service.files().list(q=query, fields="files(id, name, mimeType)").execute()
    files_to_process = results.get("files", [])
    
    if not files_to_process:
        print(f"No new document files found in user folder {user_id}")
        return []

    user_transcript_dir, _, _ = get_user_paths(user_id) 
    summary_paths = []
    
    for f in files_to_process:
        file_id = f["id"]
        file_name = f["name"]
        mime_type = f["mimeType"]
        
        processed_summary_exists = any(file_name in s for s in os.listdir(os.path.join(SUMMARY_DIR, user_id)))
        if processed_summary_exists:
             print(f"Skipping {file_name}: A summary likely exists.")
             continue
             
        # Determine export/download format
        if mime_type in DOC_MIME_TYPES:
            export_mime_type = DOC_MIME_TYPES[mime_type]
            request = service.files().export_media(fileId=file_id, mimeType=export_mime_type)
            download_filename = f"{file_name}.txt" if 'text' in export_mime_type else file_name
            
        elif mime_type == 'text/plain':
            request = service.files().get_media(fileId=file_id)
            download_filename = file_name
        else:
            print(f"Skipping {file_name}: Unsupported mimeType {mime_type}.")
            continue


        # 3. Download the file to the transcript directory
        local_transcript_path = os.path.join(user_transcript_dir, download_filename)
        fh = io.FileIO(local_transcript_path, "wb")
        downloader = MediaIoBaseDownload(fh, request)
        done = False
        while not done:
            status, done = downloader.next_chunk()
        fh.close()
        # print(f"Downloaded transcript file: {local_transcript_path}")

        # 4. Read content
        try:
            with open(local_transcript_path, "r", encoding="utf-8") as raw_f:
                raw_text = raw_f.read()
        except Exception as e:
            print(f"Error reading file {local_transcript_path}: {e}")
            os.remove(local_transcript_path)
            continue

        # 5. Generate and save the session summary
        summary_path = generate_session_summary(raw_text, download_filename, user_id)
        summary_paths.append(summary_path)
        
        # 6. Clean up the transcript file after processing
        os.remove(local_transcript_path)
        
    return summary_paths


def generate_cumulative_summary(user_id):
    _, user_summary_dir, _ = get_user_paths(user_id) 
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

    return cum_summary_path


def vectorize_cumulative_summary(user_id):
    _, user_summary_dir, user_vector_dir = get_user_paths(user_id) 
    cum_summary_path = os.path.join(user_summary_dir, "add_summary.txt")

    if not os.path.exists(cum_summary_path):
        print(f"No cumulative summary for {user_id}")
        return None

    with open(cum_summary_path, "r", encoding="utf-8") as f:
        text = f.read()

    chunks = text_splitter.split_text(text)
    db = FAISS.from_texts(chunks, embeddings)
    db.save_local(user_vector_dir, index_name="add_summary")
    return db


def load_user_vector_db(user_id):
    _, _, user_vector_dir = get_user_paths(user_id) 
    faiss_path = os.path.join(user_vector_dir, "add_summary.faiss")
    pkl_path = os.path.join(user_vector_dir, "add_summary.pkl")

    if not os.path.exists(faiss_path) or not os.path.exists(pkl_path):
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

# Pydantic models for request/response bodies
class QueryRequest(BaseModel):
    user_id: str
    query: str

class StatusResponse(BaseModel):
    status: str
    message: str
    data: Dict[str, Any] = {}
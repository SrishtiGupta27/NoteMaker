import os
import json
import time
from dotenv import load_dotenv
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_groq import ChatGroq
from pydantic import BaseModel
from typing import Dict, Any, Optional

# --- CONFIGURATION & INITIAL SETUP ---
load_dotenv()

BASE_DIR = "data"
# Transcript folder is now used for temporary storage for API 1
TRANSCRIPT_DIR = os.path.join(BASE_DIR, "transcript") 
SUMMARY_DIR = os.path.join(BASE_DIR, "summary")  
VECTOR_DIR = os.path.join(BASE_DIR, "vectors")

os.makedirs(TRANSCRIPT_DIR, exist_ok=True) 
os.makedirs(SUMMARY_DIR, exist_ok=True)
os.makedirs(VECTOR_DIR, exist_ok=True)

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
if not GROQ_API_KEY:
    raise ValueError("GROQ_API_KEY not found in .env file")

# LLM and embeddings
llm = ChatGroq(model_name="meta-llama/llama-4-maverick-17b-128e-instruct",
               groq_api_key=GROQ_API_KEY)
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)

# Stable Filenames
CUMULATIVE_SUMMARY_FILENAME = "cumulative_summary.txt"
VECTOR_INDEX_NAME = "knowledge_base"


# --- PATH MANAGEMENT ---

def get_user_paths(user_id: str) -> tuple[str, str, str]:
    """Ensures user-specific directories exist and returns their paths."""
    user_transcript_dir = os.path.join(TRANSCRIPT_DIR, user_id) 
    user_summary_dir = os.path.join(SUMMARY_DIR, user_id)
    user_vector_dir = os.path.join(VECTOR_DIR, user_id)
    
    os.makedirs(user_transcript_dir, exist_ok=True)
    os.makedirs(user_summary_dir, exist_ok=True)
    os.makedirs(user_vector_dir, exist_ok=True)
    
    return user_transcript_dir, user_summary_dir, user_vector_dir


# --- CORE LOGIC: API STAGE 1 (Summarization) ---

def generate_session_summary_from_file(raw_text_path: str, file_id: str, user_id: str) -> str:
    """Generates a summary from a local file and saves it as a JSON file."""
    _, user_summary_dir, _ = get_user_paths(user_id) 
    
    summary_filename = f"{file_id}.json"
    session_summary_path = os.path.join(user_summary_dir, summary_filename)

    # Check for duplicate
    if os.path.exists(session_summary_path):
        print(f"Summary for {file_id} already exists. Skipping summarization.")
        return session_summary_path
    
    try:
        with open(raw_text_path, "r", encoding="utf-8") as f:
            raw_text = f.read()
    except Exception as e:
        print(f"Error reading transcript file {raw_text_path}: {e}")
        return ""

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
    print(f"Generating session summary for: {file_id} for user {user_id}...")
    response = llm.invoke(prompt)
    session_summary_text = response.content.strip()

    # Save summary as JSON with metadata
    summary_data = {
        "file_id": file_id,
        "user_id": user_id,
        "summary_text": session_summary_text,
        "timestamp": time.time() # Added timestamp
    }
    
    with open(session_summary_path, "w", encoding="utf-8") as f:
        json.dump(summary_data, f, indent=4)

    print(f"Session summary saved: {session_summary_path}")
    return session_summary_path


def process_transcript_via_api(user_id: str, file_id: str, transcript_text: str) -> str:
    """
    Handles API intake, saves the raw text, generates a summary, and cleans up the raw text.
    This function is the entry point for API 1.
    """
    user_transcript_dir, _, _ = get_user_paths(user_id)
    
    # 1. Save the raw transcript to a temporary .txt file
    temp_transcript_path = os.path.join(user_transcript_dir, f"{file_id}.txt")
    
    try:
        with open(temp_transcript_path, "w", encoding="utf-8") as f:
            f.write(transcript_text)
        print(f"Raw transcript saved to: {temp_transcript_path}")
    except Exception as e:
        print(f"Error saving raw transcript: {e}")
        return f"Error: Failed to save raw transcript."

    # 2. Generate and save the session summary from the file
    summary_path = generate_session_summary_from_file(
        raw_text_path=temp_transcript_path, 
        file_id=file_id, 
        user_id=user_id
    )
    
    # 3. Clean up the temporary raw transcript file
    if os.path.exists(temp_transcript_path):
        os.remove(temp_transcript_path)
        print(f"Cleaned up raw transcript file: {temp_transcript_path}")
        
    return summary_path


# --- CORE LOGIC: API STAGE 2 (Cumulative Update & Vectorization) ---

def get_all_session_summaries(user_summary_dir: str) -> list[str]:
    """
    Loads all individual session summary texts from the .json files 
    and returns them as a single list of strings.
    """
    session_files = [
        os.path.join(user_summary_dir, f)
        for f in os.listdir(user_summary_dir)
        if f.endswith(".json")
    ]
    
    all_summaries = []
    for f in session_files:
        try:
            with open(f, "r", encoding="utf-8") as json_f:
                data = json.load(json_f)
                all_summaries.append(data.get("summary_text", ""))
        except json.JSONDecodeError:
            print(f"Warning: Could not decode JSON from {f}. Skipping.")
    
    return all_summaries


def update_and_vectorize_knowledge_base(user_id: str) -> str:
    """
    Reads ALL session summaries, generates a NEW cumulative summary, 
    deletes the OLD cumulative summary, and vectorizes the new one. 
    This is the entry point for API 2.
    """
    _, user_summary_dir, user_vector_dir = get_user_paths(user_id) 
    cum_summary_path = os.path.join(user_summary_dir, CUMULATIVE_SUMMARY_FILENAME)

    # 1. Gather ALL individual session summaries (the source of truth)
    all_session_summaries = get_all_session_summaries(user_summary_dir)
    
    if not all_session_summaries:
        return "Status: No individual session summaries found to create a knowledge base."
        
    # Combine all individual session summaries into one massive context string
    full_context_text = "\n\n---\n\n".join(all_session_summaries)
    
    # 2. Check for OLD cumulative file path (for deletion later)
    old_cum_summary_path = cum_summary_path if os.path.exists(cum_summary_path) else None

    # 3. Generate the NEW cumulative summary (LLM Call)
    # The prompt now takes ALL individual sessions as the source for the new cumulative summary
    prompt = f"""
You are an AI assistant tasked with creating a single, cohesive, and comprehensive **cumulative summary** for a user's entire history.
Below is the content from ALL recorded session summaries.

<ALL_SESSION_SUMMARIES_CONTEXT>
{full_context_text}
</ALL_SESSION_SUMMARIES_CONTEXT>

Instructions:
1. Consolidate and synthesize all information provided into one comprehensive, logical summary.
2. **Do not repeat information** unless necessary for context. Update and integrate points.
3. **Preserve all key points, decisions, action items, and important insights.**
4. Organize the summary clearly using sections or bullet points for maximum readability and ease of reference.
5. The final output must be the complete, current state of the user's knowledge base.

Output the refined cumulative summary. Do not include any text outside the summary.
"""
    print(f"Generating NEW cumulative summary for {user_id} by consolidating {len(all_session_summaries)} sessions...")
    response = llm.invoke(prompt)
    new_cum_summary = response.content.strip()

    # 4. Save the NEW cumulative summary
    with open(cum_summary_path, "w", encoding="utf-8") as f:
        f.write(new_cum_summary)

    # 5. Vectorize the new cumulative summary
    vectorize_cumulative_summary(user_id, cum_summary_path)
    
    # 6. Clean up: Delete the OLD cumulative summary file
    if old_cum_summary_path and os.path.exists(old_cum_summary_path):
        os.remove(old_cum_summary_path)
        print(f"Cleaned up OLD cumulative summary file: {old_cum_summary_path}")
        
    return f"Status: Knowledge Base Successfully Rebuilt and Vectorized for {user_id}. All session summaries ({len(all_session_summaries)} files) preserved."


def vectorize_cumulative_summary(user_id: str, cum_summary_path: str) -> None:
    """Loads the cumulative summary text, chunks it, and vectorizes it."""
    _, _, user_vector_dir = get_user_paths(user_id) 
    
    if not os.path.exists(cum_summary_path):
        raise FileNotFoundError(f"Cumulative summary file not found at {cum_summary_path}.")

    with open(cum_summary_path, "r", encoding="utf-8") as f:
        text = f.read()

    chunks = text_splitter.split_text(text)
    db = FAISS.from_texts(chunks, embeddings)
    
    # Use a stable index name for the knowledge base
    db.save_local(user_vector_dir, index_name=VECTOR_INDEX_NAME) 
    
    print(f"Vector database updated for user {user_id}.")


# --- CORE LOGIC: API STAGE 3 (QnA) ---

def load_user_vector_db(user_id: str) -> Optional[FAISS]:
    """Loads the user's vector database from the data/vectors folder."""
    _, _, user_vector_dir = get_user_paths(user_id) 
    
    INDEX_NAME = VECTOR_INDEX_NAME
    faiss_path = os.path.join(user_vector_dir, f"{INDEX_NAME}.faiss")
    pkl_path = os.path.join(user_vector_dir, f"{INDEX_NAME}.pkl")

    if not os.path.exists(faiss_path) or not os.path.exists(pkl_path):
        return None

    db = FAISS.load_local(user_vector_dir, embeddings, index_name=INDEX_NAME,
                          allow_dangerous_deserialization=True)
    return db


def get_answer_from_user_db(user_id: str, query: str) -> str:
    """Retrieves relevant context and uses the LLM to answer the user's query."""
    db = load_user_vector_db(user_id)
    if not db:
        return "No knowledge base found. Please process transcripts and update the KB first (API 1 and 2)."

    retriever = db.as_retriever(search_kwargs={"k": 5})
    docs = retriever.get_relevant_documents(query)
    context = "\n".join([d.page_content for d in docs])

    # Simplified QnA Prompt
    prompt = f"""
    You are an AI assistant helping a user based on their comprehensive session summaries.
    Answer the QUESTION using only the provided CONTEXT. If the answer is not in the context, state that you cannot answer based on the available information.

    <CONTEXT>
    {context}
    </CONTEXT>

    <QUESTION>
    {query}
    </QUESTION>

    ANSWER:
    """
    response = llm.invoke(prompt)
    return response.content.strip()


# --- Pydantic models (for API endpoints, belong in main.py) ---
class QueryRequest(BaseModel):
    user_id: str
    query: str

class TranscriptRequest(BaseModel):
    user_id: str
    file_id: str # Unique ID for the session
    transcript_text: str

class StatusResponse(BaseModel):
    status: str
    message: str
    data: Dict[str, Any] = {}
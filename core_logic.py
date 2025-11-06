import os
import json
import time
from dotenv import load_dotenv
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import PGVector
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pydantic import BaseModel
from typing import Dict, Any, Optional
from langchain_community.vectorstores.pgvector import PGVector as PGVectorType 
from langchain_openai import ChatOpenAI
from helpers.env import get_db_uri

# CONFIGURATION & INITIAL SETUP 
load_dotenv()

BASE_DIR = "./notemaker/data"
# Transcript folder is now used for temporary storage for API 1
TRANSCRIPT_DIR = os.path.join(BASE_DIR, "transcript") 
SUMMARY_DIR = os.path.join(BASE_DIR, "summary")  

os.makedirs(TRANSCRIPT_DIR, exist_ok=True) 
os.makedirs(SUMMARY_DIR, exist_ok=True)

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
POSTGRES_CONNECTION_STRING = get_db_uri()

if not OPENAI_API_KEY:
    raise ValueError("OPENAI_API_KEY not found in .env file")
if not POSTGRES_CONNECTION_STRING:
    raise ValueError("POSTGRES_CONNECTION_STRING not found in .env file")


# LLM and embeddings
llm = ChatOpenAI(model_name="gpt-4o-mini",api_key=OPENAI_API_KEY)
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)

# Stable Filenames
CUMULATIVE_SUMMARY_FILENAME = "cumulative_summary.txt"

# helpers/notemaker.py

def get_collection_name(user_id: str, service_id: str) -> str:
    """Generates a unique PGVector collection name."""
    # Collection name is built using both IDs
    return f"kb_{user_id}_{service_id}"

# PATH MANAGEMENT 

def get_user_paths(user_id: str, service_id: str) -> tuple[str, str]:
    """Ensures user-specific directories exist and returns their paths."""
    user_transcript_dir = os.path.join(TRANSCRIPT_DIR, user_id, service_id) 
    user_summary_dir = os.path.join(SUMMARY_DIR, user_id, service_id)
    # The vector dir is less relevant for PGVector but kept for consistency
   
    
    os.makedirs(user_transcript_dir, exist_ok=True)
    os.makedirs(user_summary_dir, exist_ok=True)
    
    return user_transcript_dir, user_summary_dir


# CORE LOGIC: API STAGE 1 (Summarization) 

def generate_session_summary_from_file(raw_text_path: str, file_id: str, user_id: str, service_id: str) -> tuple[str, dict[str, Any]]:
    """Generates a summary from a local file and saves it as a JSON file, or loads existing data."""
    _, user_summary_dir = get_user_paths(user_id, service_id) 
    
    summary_filename = f"{file_id}.json"
    session_summary_path = os.path.join(user_summary_dir, summary_filename)

    # Check for duplicate - FIX for "too many values to unpack" error
    if os.path.exists(session_summary_path):
        print(f"Summary for {file_id} already exists. Loading existing data.")
        try:
            with open(session_summary_path, "r", encoding="utf-8") as f:
                # Load the existing data so it can be returned as the second expected value
                existing_summary_data = json.load(f)
            # Must return a tuple of two values (path, data)
            return session_summary_path, existing_summary_data
        except Exception as e:
            print(f"Error loading existing summary file {session_summary_path}: {e}")
            # If loading fails, treat it like a new run, or raise an error
            # For this fix, we'll continue to generate a new one below, but first exit the duplicate check.
            pass # Continue to the generation logic if loading failed.
    
    # --- Code continues here for generation if file didn't exist or failed to load ---
    try:
        with open(raw_text_path, "r", encoding="utf-8") as f:
            raw_text = f.read()
    except Exception as e:
        print(f"Error reading transcript file {raw_text_path}: {e}")
        
        # FIX: Ensure a tuple of two values is returned on early exit path
        return "", {"summary_text": f"Error: Failed to read transcript file. {e}"}

    prompt = f"""

You are an AI assistant tasked with creating a **detailed and structured session summary** from a raw document or meeting transcript.
<DOCUMENT_TEXT>
{raw_text}
</DOCUMENT_TEXT>

**Instructions for Detailed Summary Generation:**

1.  Identify Core Session Details:** Extract and list the participants, session type, duration, and the overarching goal of the session.
2.  Detailed Content Analysis:**
     **Main Concern/Presenting Problem:** Detail the client's primary issue and any related background context (e.g., job stress, specific overthinking patterns).
     **Observed Symptoms/Behaviors:** Note the specific physical, emotional, or behavioral manifestations related to the main concern (e.g., tight chest, high self-rating of stress, avoidance).
     **Intervention/Technique Applied:** Describe the specific therapeutic technique or intervention introduced or practiced during the session (e.g., slow breathing, cognitive reframing).
     **Client Response/Insight:** Capture the client's immediate reaction, realization, or reported feeling change after the intervention or discussion.
3.  Key Decisions, Goals, and Action Items (Homework):** Extract and list all specific, measurable tasks or goals set for the client to work on before the next session, including any tracking or monitoring activities.
4.  Structure and Format:**
    * Organize the summary into clear, distinct sections using **markdown headings** (e.g., "Session Details," "Presenting Issue and Assessment," "Interventions and Insights," "Next Steps and Goals").
    * Use **bullet points** within each section for clarity and scannability.
5.  Tone and Purpose:** Maintain a neutral, factual, and professional tone. The summary must be detailed enough to stand as a reliable record for future cumulative summaries.

Output the detailed session summary. Do not include any text outside the summary.

"""
    print(f"Generating session summary for: {file_id} for user {user_id}...")
    response = llm.invoke(prompt)
    session_summary_text = response.content.strip()

    # Save summary as JSON with metadata
    summary_data = {
        "file_id": file_id,
        "user_id": user_id,
        "service_id": service_id,
        "summary_text": session_summary_text,
        "timestamp": time.time() # Added timestamp
    }
    
    with open(session_summary_path, "w", encoding="utf-8") as f:
        json.dump(summary_data, f, indent=4)

    print(f"Session summary saved: {session_summary_path}")
    # Must return a tuple of two values (path, data)
    return session_summary_path, summary_data





def process_transcript_via_api(user_id: str, file_id: str, transcript_text: str, service_id: str) -> tuple[str, dict[str, Any]]:
    """
    Handles API intake, saves the raw text, generates a summary, and cleans up the raw text.
    Returns the path and the summary data for API response.
    """
    user_transcript_dir, _ = get_user_paths(user_id, service_id)
    
    # 1. Save the raw transcript to a temporary .txt file
    temp_transcript_path = os.path.join(user_transcript_dir, f"{file_id}.txt")
    
    try:
        with open(temp_transcript_path, "w", encoding="utf-8") as f:
            f.write(transcript_text)
        print(f"Raw transcript saved to: {temp_transcript_path}")
    except Exception as e:
        print(f"Error saving raw transcript: {e}")
        # FIX: Ensure a tuple of two values is returned on early exit path
        return f"Error: Failed to save raw transcript.", {"summary_text": f"Error: Failed to save raw transcript. {e}"}

    # 2. Generate and save the session summary from the file
    # FIX: Capture both returned values (path, data)
    summary_path, summary_data = generate_session_summary_from_file(
        raw_text_path=temp_transcript_path, 
        file_id=file_id, 
        user_id=user_id,
        service_id=service_id
    )
    
    # 3. Clean up the temporary raw transcript file
    if os.path.exists(temp_transcript_path):
        os.remove(temp_transcript_path)
        print(f"Cleaned up raw transcript file: {temp_transcript_path}")
        
    # FIX: Return both the path and the summary data
    return summary_path, summary_data

# CORE LOGIC: API STAGE 2 (Cumulative Update & Vectorization) 

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


def update_and_vectorize_knowledge_base(user_id: str, service_id: str, session_summaries_batch: list[str]) -> str:
    """
    Accepts a batch of session summaries, consolidates them via LLM, 
    and adds the resulting consolidated document to the user's PGVector KB (appending).
    This is the entry point for API 2.
    """
    
    if not session_summaries_batch:
        return "Status: No session summaries found in the input batch to update the knowledge base."
    
    num_sessions = len(session_summaries_batch)
    batch_id = time.time() # Unique ID for this specific batch insert

    # 1. Combine the batch summaries into one context string
    full_context_text = "\n\n---\n\n".join(session_summaries_batch)
    
    # 2. Generate the NEW Consolidated Summary for THIS BATCH (LLM Call)
    prompt = f"""
You are an AI assistant tasked with creating a single, cohesive, and comprehensive **consolidated summary** from the following batch of {num_sessions} session summaries.

<SESSION_SUMMARIES_BATCH_CONTEXT>
{full_context_text}
</SESSION_SUMMARIES_BATCH_CONTEXT>

Instructions:
1. Consolidate and synthesize all information provided into one comprehensive, logical summary.
2. **Do not repeat information** unless necessary for context. Update and integrate points.
3. **Preserve all key points, decisions, action items, and important insights** from this batch.
4. The final output must be the complete, refined summary of these specific sessions.

Output the refined consolidated summary. Do not include any text outside the summary.
"""
    print(f"Generating consolidated summary for user {user_id} from a batch of {num_sessions} sessions...")
    response = llm.invoke(prompt)
    new_batch_summary = response.content.strip()

    # 3. Save the NEW consolidated summary locally (Optional Audit/Debugging)
    _, user_summary_dir = get_user_paths(user_id, service_id)
    batch_summary_filename = f"consolidated_batch_{int(batch_id)}.txt"
    cum_summary_path = os.path.join(user_summary_dir, batch_summary_filename)
    
    with open(cum_summary_path, "w", encoding="utf-8") as f:
        f.write(new_batch_summary)

    # 4. Vectorize the new summary into PGVector (APPENDING)
    vectorize_batch_for_appending(user_id, service_id, new_batch_summary, batch_id)
        
    return f"Status: Knowledge Base successfully updated. Consolidated summary from {num_sessions} sessions appended to KB for {user_id}/{service_id}."


def vectorize_cumulative_summary(user_id: str, cum_summary_path: str) -> None:
    """Loads the cumulative summary text, chunks it, and vectorizes it into PGVector."""
    
    if not os.path.exists(cum_summary_path):
        raise FileNotFoundError(f"Cumulative summary file not found at {cum_summary_path}.")

    with open(cum_summary_path, "r", encoding="utf-8") as f:
        text = f.read()

    # 1. Chunk the text
    chunks = text_splitter.split_text(text)
    
    # 2. Create documents for PGVector
    documents = [
        {"page_content": chunk, "metadata": {"user_id": user_id, "source": CUMULATIVE_SUMMARY_FILENAME}}
        for chunk in chunks
    ]
    
    # 3. Initialize PGVector with the collection name derived from user_id
    collection_name = f"kb_{user_id}"
    
    try:
        # Tries to connect and delete the collection if it exists
        temp_db = PGVector.from_existing_table(
            embedding=embeddings,
            table_name=collection_name,
            connection_string=POSTGRES_CONNECTION_STRING
        )
        temp_db.delete(filter={}) # Deletes all documents in the collection
        print(f"Cleared old vector data from PGVector collection: {collection_name}")
    except Exception as e:
        # This is expected if the collection doesn't exist yet, we can ignore it
        print(f"Attempted to clear old collection {collection_name}. May be a first run. Error: {e}")

    # Now, insert the new documents. This creates the collection if it doesn't exist.
    PGVector.from_texts(
        texts=chunks,
        embedding=embeddings,
        collection_name=collection_name,
        connection_string=POSTGRES_CONNECTION_STRING
    )
    
    print(f"PGVector database updated for user {user_id} in collection: {collection_name}.")


def vectorize_batch_for_appending(user_id: str, service_id: str, batch_text: str, batch_id: float) -> None:
    """Chunks and vectorizes the new batch text, appending it to the PGVector KB."""
    
    # 1. Chunk the text
    chunks = text_splitter.split_text(batch_text)
    
    # 2. Prepare texts and metadata
    texts = chunks
    metadatas = [
        {"user_id": user_id, "service_id": service_id, "source": f"batch_{int(batch_id)}"}
        for chunk in chunks
    ]
    
    collection_name = get_collection_name(user_id, service_id)
    
    try:
        # A. Attempt to connect to the existing PGVector collection
        db = PGVector(
            collection_name=collection_name,
            embedding_function=embeddings,
            connection_string=POSTGRES_CONNECTION_STRING,
        )
        
        # B. Append the new documents (texts) to the existing collection
        db.add_texts(texts=texts, metadatas=metadatas)
        
        print(f"PGVector database appended with new batch for user {user_id}/service {service_id}.")

    except Exception as e:
        # If the collection doesn't exist (first run), this is the fallback to create it.
        print(f"Collection {collection_name} not found. Creating new collection. Error: {e}")
        
        PGVector.from_texts(
            texts=texts,
            embedding=embeddings,
            collection_name=collection_name,
            connection_string=POSTGRES_CONNECTION_STRING,
            metadatas=metadatas
        )
        print(f"PGVector database created and populated for user {user_id}/service {service_id}.")    


# CORE LOGIC: API STAGE 3 (QnA) 

def load_user_vector_db(user_id: str, service_id: str) -> Optional[PGVectorType]:
    """Loads the user's vector database (PGVector connection) for query."""
    
    collection_name = get_collection_name(user_id, service_id)
    print(f"user_id:{user_id}, service_id:{service_id}, collection_name:{collection_name}")
    
    try:
        # Final and stable connection method: Initialize the PGVector class directly
        db = PGVector(
            collection_name=collection_name,
            embedding_function=embeddings,
            connection_string=POSTGRES_CONNECTION_STRING,
        )
        
        return db

    except Exception as e:
        # This logs any real connection failure (e.g., DB offline, bad credentials)
        print(f"Error loading PGVector for user {user_id}/{service_id}. Error: {e}")
        return None


def get_answer_from_user_db(user_id: str, service_id: str, query: str) -> str:
    """Retrieves relevant context and uses the LLM to answer the user's query."""
    db = load_user_vector_db(user_id, service_id)
    if not db:
        return "No knowledge base found. Please process transcripts and update the KB first (API 1 and 2)."

    # We use the PGVector object directly
    retriever = db.as_retriever(search_kwargs={"k": 5})

    try:
        docs = retriever.invoke(query)
    except AttributeError:
        # Fallback to direct call method if invoke() is also missing (common in older LangChain versions)
        docs = retriever(query)
    context = "\n".join([d.page_content for d in docs])

    # Simplified QnA Prompt
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

ALWAYS
- Never say that I am an AI assistant.
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


# Pydantic models (for API endpoints, belong in main.py) ---
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
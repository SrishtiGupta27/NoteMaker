from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import RedirectResponse, HTMLResponse
from typing import Dict, Any
from pydantic import BaseModel  # Standard practice: Define Pydantic models here.
from .core_logic import (
    
    
    # New Core Logic Functions 
    process_transcript_via_api,             # API Stage 1
    update_and_vectorize_knowledge_base,    # API Stage 2
    get_answer_from_user_db,                # API Stage 3
    
    # Stable Filenames (For reference, if needed) 
    CUMULATIVE_SUMMARY_FILENAME,
)

# Define Pydantic Models for Request/Response Bodies (Best practice is to define them here) 

class TranscriptRequest(BaseModel):
    user_id: str
    file_id: str # Unique ID for the session (e.g., UUID or timestamp)
    transcript_text: str

class QueryRequest(BaseModel):
    user_id: str
    query: str

class StatusResponse(BaseModel):
    status: str
    message: str
    data: Dict[str, Any] = {}

class KBUpdateRequest(BaseModel):
    user_id: str


# FastAPI Application Setup 
app = FastAPI(title="NoteMaker AI Assistant API")

# 1. Data Ingestion and Summarization Endpoint (API Stage 1) 

@app.post("/summarize", response_model=StatusResponse)
async def summarize_transcript(request: TranscriptRequest):
    """
    Ingests a raw transcript, converts it to a .txt file, generates a session summary, 
    and saves the summary JSON in the user's folder.
    """
    user_id = request.user_id
    file_id = request.file_id
    transcript_text = request.transcript_text

    if not user_id or not file_id:
        raise HTTPException(status_code=400, detail="User ID and File ID are required.")

    try:
        # Calls the core logic to process the raw text and save the summary
        summary_path = process_transcript_via_api(user_id, file_id, transcript_text)
        
        return StatusResponse(
            status="success", 
            message="Transcript processed and individual session summary saved.",
            data={"session_summary_path": summary_path}
        )
    except Exception as e:
        # NOTE: If this fails, check your GROQ_API_KEY environment variable and API access.
        raise HTTPException(status_code=500, detail=f"Summarization failed: {e}")


# 2. Knowledge Base Update Endpoint (API Stage 2) 

@app.post("/update-kb", response_model=StatusResponse)
async def update_knowledge_base(request: KBUpdateRequest):
    """
    Consolidates ALL session summaries, generates a new cumulative summary, 
    **vectorizes it using PGVector**, and updates the knowledge base.
    """
    user_id = request.user_id

    if not user_id:
        raise HTTPException(status_code=400, detail="User ID is required.")

    try:
        # Calls the core logic to update the cumulative summary and vector store (now PGVector)
        status_message = update_and_vectorize_knowledge_base(user_id)
        
        if "No summaries" in status_message:
             return StatusResponse(
                status="warning", 
                message=status_message,
                data={}
            )

        return StatusResponse(
            status="success", 
            message=status_message,
            data={"cumulative_summary_filename": CUMULATIVE_SUMMARY_FILENAME}
        )
    except Exception as e:
        # NOTE: This can fail due to LLM errors, file system access, or **PostgreSQL connection/permissions**.
        raise HTTPException(status_code=500, detail=f"Knowledge Base update failed: {e}")


# 3. Query Endpoint (API Stage 3) 

@app.post("/query", response_model=StatusResponse)
async def query_assistant(request: QueryRequest):
    """Answers a question based on the user's vectorized cumulative summary (now stored in PGVector)."""
    user_id = request.user_id
    query = request.query

    if not user_id or not query:
        raise HTTPException(status_code=400, detail="User ID and query are required.")

    try:
        answer = get_answer_from_user_db(user_id, query)
        
        # This check covers the case where the PGVector collection does not yet exist for the user
        if "No knowledge base found" in answer:
            return StatusResponse(
                status="warning", 
                message=answer,
                data={"answer": answer}
            )

        return StatusResponse(
            status="success", 
            message="Query processed.",
            data={"answer": answer}
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Query failed: {e}")
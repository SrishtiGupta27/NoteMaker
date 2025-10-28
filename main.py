from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import RedirectResponse, HTMLResponse
from typing import Dict, Any
from core_logic import (
    get_auth_flow, complete_authorization, get_drive_service_from_token,
    fetch_and_process_session_files, generate_cumulative_summary, 
    vectorize_cumulative_summary, get_answer_from_user_db, 
    QueryRequest, StatusResponse
)

app = FastAPI(title="NoteMaker AI Assistant API")

# Temporary storage for the Google OAuth flow object
# IMPORTANT: In a production setting, replace this with a persistent store (e.g., Redis)
OAUTH_FLOW_STATE: Dict[str, Any] = {}

# --- 1. Authorization Endpoints ---

@app.get("/authorize")
async def authorize(request: Request):
    """Starts the Google Drive OAuth flow."""
    # The callback URL must be exactly http://127.0.0.1:8000/google-oauth-callback (or your public URL)
    callback_url = str(request.url).replace("/authorize", "/google-oauth-callback")
    
    try:
        flow = get_auth_flow(request_url=callback_url)
        auth_url, state = flow.authorization_url(prompt='consent')
        
        # Store the flow object
        OAUTH_FLOW_STATE[state] = flow
        
        return RedirectResponse(auth_url)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Authorization setup failed. Ensure .credentials/credentials.json is correct: {e}")

@app.get("/google-oauth-callback")
async def google_oauth_callback(state: str, code: str = None):
    """Receives the authorization code from Google and completes the process."""
    if not code:
        return StatusResponse(status="error", message="Authorization denied by user.")

    if state not in OAUTH_FLOW_STATE:
        raise HTTPException(status_code=400, detail="Invalid state parameter. Authorization flow lost.")

    flow = OAUTH_FLOW_STATE.pop(state) # Get and remove the flow object
    
    try:
        complete_authorization(flow, code)
        return StatusResponse(status="success", message="Google Drive authorized successfully and token saved to .credentials/token.json.")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Token exchange failed: {e}")


# --- 2. Sync/Update Endpoint ---

@app.post("/sync-data/{user_id}", response_model=StatusResponse)
async def sync_data(user_id: str):
    """Fetches documents, generates session summaries, combines into cumulative summary, and vectorizes."""
    if not user_id:
        raise HTTPException(status_code=400, detail="User ID is required.")

    service = get_drive_service_from_token()
    if not service:
        raise HTTPException(status_code=401, detail="Google Drive not authorized. Please run /authorize first.")

    try:
        # 1. Fetch, convert, and generate individual session summaries
        summary_paths = fetch_and_process_session_files(service, user_id)
        
        # 2. Generate cumulative summary from new session summaries
        cum_summary_path = generate_cumulative_summary(user_id)
        
        # 3. Vectorize the final cumulative summary
        vectorize_cumulative_summary(user_id)
        
        return StatusResponse(
            status="success", 
            message="Data sync, summarization, and vectorization completed.",
            data={"new_sessions_processed": len(summary_paths), "cumulative_summary_path": cum_summary_path}
        )
    except Exception as e:
        # NOTE: A failure here could be due to API limits, file format errors, or LLM issues
        raise HTTPException(status_code=500, detail=f"Data sync failed: {e}")


# --- 3. Query Endpoint ---

@app.post("/query", response_model=StatusResponse)
async def query_assistant(request: QueryRequest):
    """Answers a question based on the user's vectorized summary."""
    user_id = request.user_id
    query = request.query

    if not user_id or not query:
        raise HTTPException(status_code=400, detail="User ID and query are required.")

    try:
        answer = get_answer_from_user_db(user_id, query)
        
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
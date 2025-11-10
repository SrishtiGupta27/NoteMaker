# NoteMaker AI Module

NoteMaker is a powerful, self-contained module designed to function as a secure and personalized long-term memory for AI assistants. Its primary purpose is to ingest raw session transcripts, process them into structured summaries, and store them in a user-specific vector knowledge base. This allows an AI assistant, like MantraAssistant, to retrieve relevant historical context, ensuring that its responses are deeply personalized, consistent, and aware of the user's journey over time.

The module is built with FastAPI and LangChain, using PostgreSQL with PGVector as its storage backend.

<img src="https://github.com/user-attachments/assets/6423fdcb-2202-46af-8796-846ac93276ed" width="500" height="700">


<br>
## Core Architecture & Workflow

NoteMaker operates on a multi-stage workflow, exposed through a series of API endpoints. This design separates the concerns of data ingestion, knowledge base construction, and querying.

### Stage 1: Individual Session Summarization

1.  A raw transcript from a single session is processed.
2.  A powerful LLM (gpt-4o-mini) is used with a detailed prompt to generate a structured, professional summary of the session. This captures key issues, interventions, and client responses.

### Stage 2: Knowledge Base Consolidation & Update

1.  Instead of vectorizing every individual summary, this stage accepts a batch of summaries.
2.  It uses an LLM to intelligently combine these summaries into a single, cohesive, and de-duplicated document. This creates a more refined and contextually rich source of truth.
3.  This new consolidated summary is then converted into vector embeddings and appended to the user's personal collection in the PGVector database.

### Stage 3: Contextual Querying

1.  When a query is made, the module performs a vector similarity search against the user's specific knowledge base.
2.  The most relevant document chunks are retrieved and used as context for an LLM, which then generates a natural language answer based on the user's history.

## Key Technologies

-   **Framework**: FastAPI
-   **AI & Orchestration**: LangChain
-   **Language Model**: OpenAI gpt-4o-mini
-   **Embeddings**: HuggingFace all-MiniLM-L6-v2
-   **Vector Database**: PostgreSQL with the PGVector extension.

## API Endpoints

The module exposes three primary endpoints to manage the workflow.

### 1. Summarize Transcript

`POST /notemaker/summarize`

This endpoint ingests a raw transcript for a single session and returns a structured summary. It's the first step in the data ingestion pipeline.

**Purpose**: To convert unstructured text from a session into a high-quality, structured summary.

**Request Body**:

```json
{
  "user_id": "string",
  "service_id": "string",
  "file_id": "string",
  "transcript_text": "string"
}
```

**Successful Response**:

```json
{
  "status": "success",
  "message": "Transcript processed and individual session summary saved.",
  "data": {
    "session_summary_path": "./notemaker/data/summary/...",
    "summary_text": "## Session Details...\\n- Goal: ...\\n\\n## Presenting Issue..."
  }
}
```

### 2. Update Knowledge Base

`POST /notemaker/update-kb`

This endpoint takes a batch of individual summaries (generated from Stage 1), consolidates them into a single document, and vectorizes it into the user's knowledge base.

**Purpose**: To intelligently combine multiple session summaries and update the user's long-term memory in the vector store.

**Request Body**:

```json
{
  "user_id": "string",
  "service_id": "string",
  "session_summaries": [
    "string"
  ]
}
```

**Successful Response**:

```json
{
  "status": "success",
  "message": "Status: Knowledge Base successfully updated. Consolidated summary from 1 sessions appended to KB for 123/456.",
  "data": {
    "cumulative_summary_filename": "cumulative_summary.txt"
  }
}
```

### 3. Query Assistant

`POST /notemaker/query`

This endpoint answers a question by retrieving context from a user's personal knowledge base.

**Purpose**: To ask questions against a user's history and get a context-aware answer.

**Request Body**:

```json
{
  "user_id": "string",
  "service_id": "string",
  "query": "string"
}
```

**Successful Response**:

```json
{
  "status": "success",
  "message": "Query processed.",
  "data": {
    "answer": "Based on our previous sessions, we discussed using breathing exercises to manage anxiety before your presentation..."
  }
}
```

## Integration with MantraAssistant

The MantraAssistant leverages this module to provide proactive, context-aware responses. In its LangGraph workflow, a dedicated node (`__inject_patient_history`) calls the `get_answer_from_user_db` function at the start of every turn. This ensures the assistant is always equipped with the user's relevant history before it even begins to formulate a reply, making for a seamless and intelligent user experience.

Here’s an updated **README** tailored for your **Streamlit version** with vectorized knowledge base and Q&A functionality:

---

# Meeting Summarizer & Transcript Q&A with Google Drive Integration

## Overview

This Python project fetches Google Docs from a Google Drive folder, converts them to text, and optionally summarizes them. It also builds **FAISS embeddings** for semantic search, allowing you to **ask questions interactively across all transcripts** using a **Groq LLM**.

The system ensures confidentiality by **never revealing personal names or identifiers** from transcripts.

---

## Features

* ✅ Fetch Google Docs from a Google Drive folder (hard-coded folder ID)
* ✅ Convert Google Docs to `.txt` locally
* ✅ Only fetches new files; skips already saved transcripts
* ✅ Creates **FAISS embeddings** for semantic search
* ✅ Maintains a **vectorized knowledge base** for all transcripts
* ✅ Interactive **Q&A interface** using Streamlit
* ✅ Strictly answers based on the knowledge base; no personal names shown
* ✅ Error handling for missing credentials, empty folders, or permissions issues

---

## Folder Structure

```
project/
│
├─ app.py      # Streamlit app script
├─ .env                   # Environment variables (GROQ_API_KEY)
├─ .credentials/          # Google API credentials folder
│    ├─ credentials.json
│    └─ token.json
├─ data/
│    ├─ transcripts/      # Fetched Google Docs as .txt
│    └─ vectors/          # FAISS vector stores for transcripts
└─ README.md
```

---

## Requirements

* Python 3.10+
* Pip packages (install via `pip install -r requirements.txt`):

```
streamlit
langchain
langchain-community
langchain-groq
python-dotenv
google-api-python-client
faiss-cpu
```

---

## Setup

### 1. Google Drive API Credentials

1. Go to [Google Cloud Console](https://console.cloud.google.com/).
2. Enable **Google Drive API** for your project.
3. Create **OAuth 2.0 credentials** and download `credentials.json`.
4. Place `credentials.json` in `.credentials/` folder.

> The app will generate `token.json` automatically after the first authentication.

---

### 2. Environment Variables

Create a `.env` file in the project root:

```
GROQ_API_KEY=your_groq_api_key_here
```

---

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

Or individually:

```bash
pip install streamlit langchain langchain-community langchain-groq python-dotenv google-api-python-client faiss-cpu
```

---

## Usage

### 1. Run the Streamlit App

```bash
streamlit run app2_streamlit.py
```

The web interface has three steps:

1. **Fetch new Google Docs** – Downloads only new files from the Drive folder.
2. **Build/Load Vectors** – Creates FAISS embeddings for transcripts not yet vectorized.
3. **Ask Questions** – Enter questions and receive answers based strictly on the knowledge base.

---

### 2. Transcript Storage

* `data/transcripts/` → Stores raw transcript `.txt` files
* `data/vectors/` → Stores FAISS vector files for each transcript

---

### 3. Sample Q&A Interaction

**Question:**

> What were the main strategies recommended in the last session?

**Answer (from vector knowledge base):**

```
- Strategy 1: Implement a weekly progress check
- Strategy 2: Schedule follow-up sessions for accountability
```

> Names or personal identifiers are never revealed.

---

## Customization

### 1. Prompt Template

Modify the LLM prompt in `get_answer_from_db()` to change:

* Tone (professional, casual, friendly)
* Instructions for handling context and restrictions

### 2. Chunk Size

```python
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
```

Adjust `chunk_size` and `chunk_overlap` to control document chunking for FAISS embedding.

### 3. FAISS Retrieval

* `k=5` → Number of chunks retrieved per query
* Increase for more context, reduce for faster answers

---

## Error Handling

* **No folder ID / invalid folder** → Script skips fetch
* **Empty folder** → No transcripts saved
* **Missing credentials** → OAuth flow will fail
* **Private folder / permission issues** → Ensure folder is shared with your OAuth account

---

## Tips

* Test with a small Drive folder first
* Use descriptive file names in Google Drive for clean transcript names
* Only text-based Google Docs are fully supported (Sheets/Slides may not render correctly)

---

Do you want me to also **update the README with screenshots and Streamlit UI instructions** for a more professional look?

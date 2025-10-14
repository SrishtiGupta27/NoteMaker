# Meeting Summarizer with Google Drive Integration

## Overview

This Python project automatically fetches Google Docs from a Google Drive folder, converts them to text, and summarizes them into structured meeting notes using **RAG (retrieval-augmented generation)** with a **Groq LLM**.

The summaries follow a professional, neutral, and concise format, ideal for therapy, coaching, mentoring, or consulting session transcripts.

---

## Features

* ✅ Fetch all Google Docs from a Google Drive folder (supports folder **links or IDs**)
* ✅ Converts Google Docs to `.txt` files locally
* ✅ Chunking with **RecursiveCharacterTextSplitter** for large transcripts
* ✅ Creates **FAISS embeddings** for semantic search
* ✅ Summarizes meetings using **Groq LLM** and a customizable prompt template
* ✅ Saves summaries in a dedicated folder
* ✅ Error handling for missing credentials, empty folders, or invalid links

---

## Folder Structure

```
project/
│
├─ app.py                 # Main script
├─ .env                   # Environment variables (GROQ_API_KEY)
├─ .credentials/          # Google API credentials folder
│    ├─ credentials.json
│    └─ token.json
├─ data/
│    ├─ transcripts/      # Fetched Google Docs as .txt
│    └─ summary/          # Generated summaries
└─ README.md
```

---

## Requirements

* Python 3.10+
* Pip packages (install via `pip install -r requirements.txt`):

```
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

> The script will automatically generate `token.json` after the first authentication.

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
pip install langchain langchain-community langchain-groq python-dotenv google-api-python-client faiss-cpu
```

---

## Usage

### 1. Provide Google Drive Folder

You can use either a **folder link** or **folder ID**:

```python
folder_link = "https://drive.google.com/drive/folders/1KtbksA2D6I2cFzplbxfahN2ZVnnTlstb?usp=sharing"
```

### 2. Run the Script

```bash
python app.py
```

The script will:

1. Fetch all Google Docs from the folder
2. Save them as `.txt` in `data/transcripts/`
3. Chunk the transcripts for semantic retrieval
4. Summarize each transcript with **Groq LLM**
5. Save summaries in `data/summary/`

---

### 3. Output Example

* `data/transcripts/Session_1.txt` → raw transcript
* `data/summary/Session_1_summary.txt` → generated meeting summary

Sample summary structure:

```
Key Topics Discussed
- Topic 1
- Topic 2

Insights or Advice Shared by the Expert
- Key insight 1
- Key insight 2

Actions or Strategies Recommended
- Action 1
- Action 2

Next Steps / Follow-up Items
- Step 1
- Step 2
```

---

## Customization

### 1. Prompt Template

Modify `custom_prompt` in `app.py` to change:

* Tone (professional, casual, friendly, etc.)
* Sections included
* Instruction style (bullet points, tables, etc.)

---

### 2. Chunk Size

```python
splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
```

Adjust `chunk_size` and `chunk_overlap` to control document chunking for FAISS embedding.

---

### 3. FAISS Retrieval

* `k=5` → Number of chunks retrieved for summarization.
* Increase for more context or reduce for faster processing.

---

## Error Handling

* ❌ **No folder ID / invalid link** → Script skips fetch
* ⚠️ **Folder empty** → Script halts with warning
* ❌ **Credentials missing** → Script raises error
* ❌ **Private folder / permissions issue** → Google API will fail; ensure folder is shared with your OAuth account

---

## Tips

* Always test with a **small folder** first.
* Ensure Google Docs are **text-based** (Sheets/Slides may not render correctly).
* Use descriptive file names in Google Drive for clean summary filenames.

---


import os
from dotenv import load_dotenv
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
from langchain_google_community import GoogleDriveLoader
from langchain_groq import ChatGroq


# Custom Prompt Template

custom_prompt = """
<task>
You are an AI meeting summarizer.
Your goal is to generate clear, structured notes from a transcript of an expert-led session (therapy, coaching, mentoring, or consulting).
Keep the tone professional, neutral, and concise.
Do NOT create tables, numbered lists, or conversational dialogue.
Use only headings and bullet points for clarity.
</task>

<structure>
- Key Topics Discussed
- Insights or Advice Shared by the Expert
- Emotional or Behavioral Patterns Observed (if applicable)
- Actions or Strategies Recommended
- Progress or Outcomes Since Last Session
- Next Steps / Follow-up Items
</structure>

<instructions>
1. Write in short, professional bullet points.
2. Maintain objectivity — avoid assumptions or personal opinions.
3. Keep each bullet informative but under two lines.
4. Remove timestamps and speaker names.
5. Avoid filler phrases like “In conclusion” or “The expert said”.
6. If a section doesn’t apply, skip it (don’t leave empty headings).
</instructions>

<output_format>
Headings: Title case (e.g., "Key Topics Discussed")
Bullets: Start with a capital letter, no full stop at the end.
Use line breaks between sections for readability.
</output_format>
"""

# Load Environment Variables

load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

if not GROQ_API_KEY:
    raise ValueError(" GROQ_API_KEY not found. Add it to your .env file.")


# Fetch All Google Docs From a Folder

def fetch_all_docs_from_folder(folder_id: str):
    """
    Fetches all Google Docs from a specific Google Drive folder,
    converts each to text, and saves them in the data/transcripts folder.
    """

    print(f" Fetching all Google Docs from folder ID: {folder_id}")
    if not folder_id:
        print(" No folder ID provided. Skipping fetch.")
        return 0
    else:
        print("Success!!")

# https://drive.google.com/drive/folders/1KtbksA2D6I2cFzplbxfahN2ZVnnTlstb?usp=sharing

    # Ensure credentials path is absolute and directory exists
    creds_path = os.path.abspath(".credentials/credentials.json")
    token_path = os.path.abspath(".credentials/token.json")
    os.makedirs(os.path.dirname(creds_path), exist_ok=True)

    # Create loader (auto-uses OAuth credentials or service account)
    loader = GoogleDriveLoader(
        folder_id=folder_id,
        recursive=True,  # Set True if you want subfolders too
        credentials_path=creds_path,  # path to your credentials file
        token_path=token_path,  # path to your token file
        load_auth=True,  # Use OAuth flow with credentials.json
    )
    docs = loader.load()
    
    print(f"Documents loaded: {len(docs)}")
    for doc in docs:
        print(f"  - {doc.metadata.get('name')} (type: {doc.metadata.get('mimeType')})")

        if not docs:
            print(" No documents found in the specified Google Drive folder.")
            print(" Please check the folder ID and ensure it contains supported file types (Google Docs, Sheets, Presentations).")
            return 0



    # Ensure folder exists
    transcripts_folder = "data/transcripts"
    os.makedirs(transcripts_folder, exist_ok=True)

    # Save each doc as .txt
    for i, doc in enumerate(docs):
        safe_title = doc.metadata.get("name", f"doc_{i+1}")
        filename = f"{safe_title.replace(' ', '_').replace('/', '_')}.txt"
        output_path = os.path.join(transcripts_folder, filename)

 
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(doc.page_content)

        print(f"Saved {filename} ({len(doc.page_content)} chars)")

    print("All Google Docs fetched and saved to data/transcripts/")
    return len(docs)

# Summarize One Transcript

def summarize_meeting(file_path: str, prompt_template: str):
    """Summarizes a single transcript file using RAG + Groq LLM."""
    with open(file_path, "r", encoding="utf-8") as f:
        text = f.read()

    # Split transcript into smaller chunks
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    chunks = splitter.split_text(text)
    docs = [Document(page_content=chunk) for chunk in chunks]

    # Create embeddings and FAISS index
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    db = FAISS.from_documents(docs, embeddings)
    retriever = db.as_retriever(search_kwargs={"k": 5})

    # Retrieve relevant chunks
    relevant_docs = retriever.get_relevant_documents("meeting summary")
    context = "\n\n".join([d.page_content for d in relevant_docs])

    # Initialize Groq LLM
    llm = ChatGroq(model_name="mixtral-8x7b", groq_api_key=GROQ_API_KEY)

    # Build summarization prompt
    final_prompt = f"""
You are an expert meeting summarizer.

Context:
{context}

Task:
{prompt_template}
"""

    # Generate the summary
    result = llm.invoke(final_prompt)
    return result.content


# Main Runner

if __name__ == "__main__":
    transcripts_folder = "data/transcripts"
    summary_folder = "data/summary"

    os.makedirs(transcripts_folder, exist_ok=True)
    os.makedirs(summary_folder, exist_ok=True)

    # Optional: Fetch from Google Drive Folder (first step)
    # Replace this ID with your Drive folder’s ID
    folder_link = "https://drive.google.com/drive/folders/1KtbksA2D6I2cFzplbxfahN2ZVnnTlstb?usp=sharing"
    fetch_all_docs_from_folder(folder_link)

    docs_fetched_count = fetch_all_docs_from_folder(folder_link)

    # If no documents were fetched, stop the script.
    if docs_fetched_count == 0:
        print("\nHalting script because no documents were fetched.")
        exit()

    # Then summarize all fetched transcripts
    for filename in os.listdir(transcripts_folder):
        if filename.endswith(".txt"):
            file_path = os.path.join(transcripts_folder, filename)
            print(f"\n Summarizing {filename} ...")

            summary = summarize_meeting(file_path, custom_prompt)

            # Print summary to console
            print(f"\n--- Meeting Summary for {filename} ---\n")
            print(summary)

            # Save summary
            summary_filename = f"{os.path.splitext(filename)[0]}_summary.txt"
            summary_path = os.path.join(summary_folder, summary_filename)
            with open(summary_path, "w", encoding="utf-8") as f:
                f.write(summary)
            print(f" Saved summary to {summary_path}")

    print("\n All summaries generated successfully!")

# https://drive.google.com/drive/folders/1KtbksA2D6I2cFzplbxfahN2ZVnnTlstb?usp=sharing
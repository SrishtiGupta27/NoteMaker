import os
from langchain_google_community import GoogleDriveLoader

# Test script to debug Google Drive access
folder_id = "1KtbksA2D6I2cFzplbxfahN2ZVnnTlstb"

creds_path = os.path.abspath(".credentials/credentials.json")
token_path = os.path.abspath(".credentials/token.json")

print(f"🔍 Checking credentials...")
print(f"  Credentials file: {creds_path}")
print(f"  Credentials exists: {os.path.exists(creds_path)}")
print(f"  Token file: {token_path}")
print(f"  Token exists: {os.path.exists(token_path)}")

if not os.path.exists(creds_path):
    print("\n❌ Credentials file not found!")
    print("   Download it from Google Cloud Console → Service Accounts")
    exit()

try:
    print(f"\n📂 Attempting to load from folder: {folder_id}")
    
    loader = GoogleDriveLoader(
        folder_id=folder_id,
        recursive=True,
        credentials_path=creds_path,
        token_path=token_path,
        load_auth=True,
    )
    
    print("✅ Loader initialized successfully")
    
    docs = loader.load()
    
    print(f"✅ Found {len(docs)} documents")
    
    if docs:
        for i, doc in enumerate(docs):
            print(f"\n  Doc {i+1}:")
            print(f"    Name: {doc.metadata.get('name')}")
            print(f"    MIME Type: {doc.metadata.get('mimeType')}")
            print(f"    Content length: {len(doc.page_content)} chars")
    else:
        print("\n⚠️ No documents found. Possible causes:")
        print("   1. Folder is empty")
        print("   2. Folder not shared with service account email")
        print("   3. Folder contains non-Google Doc files")
        print("   4. API permissions not enabled")
        
except Exception as e:
    print(f"\n❌ Error: {type(e).__name__}")
    print(f"   Message: {e}")
    print("\n💡 Common issues:")
    print("   - Service account email not added to folder sharing")
    print("   - Google Drive API not enabled in Cloud Console")
    print("   - Credentials file is invalid or expired")
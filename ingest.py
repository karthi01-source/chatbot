import os
os.environ['FAISS_OPT_LEVEL'] = 'generic'
from dotenv import load_dotenv
load_dotenv()
import pickle
# from langchain_text_splitters import RecursiveCharacterTextSplitter # REMOVED
import faiss
import requests
import json
import time
import numpy as np
import pypdf

print("Script started...")

# --- 1. Configuration ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
VECTOR_DB_PATH = os.path.join(BASE_DIR, "faiss_index")
DATA_STORE_PATH = os.path.join(BASE_DIR, "data_store.pkl")
CACHE_PATH = os.path.join(BASE_DIR, "embeddings_cache.pkl")
MODEL_NAME = 'all-MiniLM-L6-v2'

# --- IMPORTANT! PASTE YOUR KEY HERE (Or use env var) ---
API_KEY = os.getenv("GEMINI_API_KEY")

def get_embedding(text):
    if not API_KEY:
        print("Error: GEMINI_API_KEY not set in .env file.")
        return None

    url = f"https://generativelanguage.googleapis.com/v1beta/models/text-embedding-004:embedContent?key={API_KEY}"
    payload = {
        "model": "models/text-embedding-004",
        "content": {
            "parts": [{"text": text}]
        }
    }
    headers = {'Content-Type': 'application/json'}
    try:
        response = requests.post(url, headers=headers, data=json.dumps(payload), timeout=15)
        if response.status_code == 200:
            return response.json()['embedding']['values']
        else:
            print(f"Error getting embedding: {response.text}")
            return None
    except Exception as e:
        print(f"Error: {e}")
        return None

def get_embeddings_batch(texts):
    """
    Get embeddings for a list of texts in a single batch call.
    Max 100 items per batch.
    """
    if not texts:
        return []
    
    url = f"https://generativelanguage.googleapis.com/v1beta/models/text-embedding-004:batchEmbedContents?key={API_KEY}"
    
    requests_payload = []
    for text in texts:
        requests_payload.append({
            "model": "models/text-embedding-004",
            "content": {
                "parts": [{"text": text}]
            }
        })
        
    payload = {"requests": requests_payload}
    headers = {'Content-Type': 'application/json'}
    
    try:
        response = requests.post(url, headers=headers, data=json.dumps(payload), timeout=30)
        if response.status_code == 200:
            results = response.json().get('embeddings', [])
            return [res['values'] for res in results]
        else:
            print(f"Error in batch embedding: {response.text}")
            return [None] * len(texts)
    except Exception as e:
        print(f"Error during batch call: {e}")
        return [None] * len(texts)

# --- Custom Text Splitter ---
def split_text_recursive(text, chunk_size=1000, chunk_overlap=200):
    """
    Splits text into chunks recursively, trying to break at paragraphs, then sentences, etc.
    This is a simplified version of RecursiveCharacterTextSplitter.
    """
    separators = ["\n\n", "\n", " ", ""]
    chunks = []
    start = 0
    length = len(text)

    while start < length:
        end = min(start + chunk_size, length)
        
        if end < length:
            # Try to find a separator to break at
            for sep in separators:
                if sep == "":
                    break # Just break at chunk_size
                
                # Look for the last occurrence of the separator within the overlap window
                # We want to keep the chunk size close to chunk_size, but not exceed it
                # and ideally break at a natural boundary.
                
                # Search backwards from 'end'
                sep_index = text.rfind(sep, start, end)
                
                if sep_index != -1 and sep_index > start:
                    end = sep_index + len(sep)
                    break
        
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        
        # Move start forward, subtracting overlap
        if end == length:
            break
            
        new_start = end - chunk_overlap
        
        # Ensure we always make progress and don't get stuck
        if new_start <= start:
             start = end 
        else:
             start = new_start

    return chunks

def extract_text_from_pdf(filepath):
    """
    Extracts plain text from a PDF file using pypdf.
    """
    text = ""
    try:
        reader = pypdf.PdfReader(filepath)
        total_pages = len(reader.pages)
        print(f"Reading {total_pages} pages from {os.path.basename(filepath)}...")
        for i, page in enumerate(reader.pages):
            if i % 10 == 0:
                 print(f"  - Progress: {i}/{total_pages} pages...")
            content = page.extract_text()
            if content:
                text += content + "\n"
        print(f"✅ Successfully extracted {len(text)} characters from {os.path.basename(filepath)}.")
        return text
    except Exception as e:
        print(f"❌ Error extracting text from PDF {filepath}: {e}")
        return ""

def rebuild_brain(upload_dir="data_uploads"):
    print(f"Starting brain rebuild from directory: {upload_dir}")

    if not os.path.exists(upload_dir):
        print(f"Error: Directory not found: {upload_dir}")
        return False

    all_chunks = []
    
    # --- 2. Iterate through all .txt and .pdf files in the directory ---
    supported_extensions = ('.txt', '.pdf')
    files = [f for f in os.listdir(upload_dir) if f.lower().endswith(supported_extensions)]
    if not files:
        print(f"No supported files ({supported_extensions}) found to ingest.")
        # Clear the index and data store if no files are present
        if os.path.exists(VECTOR_DB_PATH):
            os.remove(VECTOR_DB_PATH)
        if os.path.exists(DATA_STORE_PATH):
            os.remove(DATA_STORE_PATH)
        print("Knowledge base cleared.")
        return True

    for filename in files:
        file_path = os.path.join(upload_dir, filename)
        print(f"Processing file: {file_path}...")
        try:
            raw_text = ""
            if filename.lower().endswith('.txt'):
                with open(file_path, 'r', encoding='utf-8') as f:
                    raw_text = f.read()
            elif filename.lower().endswith('.pdf'):
                raw_text = extract_text_from_pdf(file_path)

            if raw_text.strip():
                # --- 3. Split Text into Chunks ---
                chunks = split_text_recursive(raw_text, chunk_size=1000, chunk_overlap=200)
                all_chunks.extend(chunks)
                print(f"✅ Added {len(chunks)} chunks from {filename} (Total chunks: {len(all_chunks)})")
            else:
                print(f"⚠️ Warning: No text extracted from {filename}.")
        except Exception as e:
            print(f"Error reading {filename}: {e}")

    if not all_chunks:
        print("Error: No text found in any of the files.")
        return False

    # --- 5. Create Embeddings with Cache ---
    print(f"Loading/Updating embeddings for total {len(all_chunks)} chunks...")
    
    # Load cache
    cache = {}
    if os.path.exists(CACHE_PATH):
        try:
            with open(CACHE_PATH, 'rb') as f:
                cache = pickle.load(f)
            print(f"Loaded {len(cache)} entries from cache.")
        except Exception as e:
            print(f"Error loading cache: {e}")

    embeddings_map = {}
    chunks_to_embed = []
    
    # Identify which chunks need new embeddings
    for chunk in all_chunks:
        if chunk in cache:
            embeddings_map[chunk] = cache[chunk]
        else:
            if chunk not in chunks_to_embed: # Avoid duplicates in the same batch
                chunks_to_embed.append(chunk)

    # Process new chunks in batches
    batch_size = 100
    new_embeddings_count = 0
    
    if chunks_to_embed:
        print(f"Requesting embeddings for {len(chunks_to_embed)} new chunks in batches of {batch_size}...")
        for i in range(0, len(chunks_to_embed), batch_size):
            batch = chunks_to_embed[i:i + batch_size]
            print(f"Processing batch {i//batch_size + 1}/{(len(chunks_to_embed)-1)//batch_size + 1}...")
            
            batch_results = get_embeddings_batch(batch)
            
            for chunk, emb in zip(batch, batch_results):
                if emb:
                    embeddings_map[chunk] = emb
                    cache[chunk] = emb
                    new_embeddings_count += 1
            
            # Rate limiting for batches
            if i + batch_size < len(chunks_to_embed):
                time.sleep(1.0) 

    # Final ordered embeddings list
    embeddings = []
    for chunk in all_chunks:
        if chunk in embeddings_map:
            embeddings.append(embeddings_map[chunk])
        else:
             # This should ideally not happen if batch calls succeed
             pass

    if not embeddings:
        print("Error: No embeddings were created.")
        return False

    # Save cache if updated
    if new_embeddings_count > 0:
        print(f"Saving {len(cache)} entries to cache...")
        try:
            with open(CACHE_PATH, 'wb') as f:
                pickle.dump(cache, f)
        except Exception as e:
            print(f"Error saving cache: {e}")

    embeddings = np.array(embeddings).astype('float32')
    print(f"Total embeddings prepared: {embeddings.shape} ({new_embeddings_count} new, {len(all_chunks) - new_embeddings_count} from cache)")

    # --- 6. Create and Save FAISS Vector Database ---
    print("Creating FAISS index...")
    d = embeddings.shape[1] 
    index = faiss.IndexFlatL2(d)
    index.add(embeddings)

    print(f"Saving FAISS index to: {VECTOR_DB_PATH}")
    faiss.write_index(index, VECTOR_DB_PATH)

    # --- 7. Save the Text Chunks ---
    print(f"Saving text chunks to: {DATA_STORE_PATH}...")
    with open(DATA_STORE_PATH, 'wb') as f:
        pickle.dump(all_chunks, f)

    print("-" * 30)
    print("✅ Brain Rebuild Complete!")
    print(f"Your updated AI brain is saved in '{VECTOR_DB_PATH}' and '{DATA_STORE_PATH}'.")
    print("-" * 30)
    return True

if __name__ == "__main__":
    rebuild_brain()

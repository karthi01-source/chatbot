import os
import pickle
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
import faiss

print("Script started...")

# --- 1. Configuration ---
# Make sure this points to your new DAA notes file
TEXT_FILE_PATH = "handbook.txt"  
VECTOR_DB_PATH = "faiss_index"
DATA_STORE_PATH = "data_store.pkl"
MODEL_NAME = 'all-MiniLM-L6-v2'

# Check if text file exists
if not os.path.exists(TEXT_FILE_PATH):
    print(f"Error: Text file not found at {TEXT_FILE_PATH}")
    print("Please make sure 'daa_notes.txt' is in the same folder.")
    exit()

# --- 2. Load and Read Text File ---
print(f"Loading text file: {TEXT_FILE_PATH}...")
try:
    with open(TEXT_FILE_PATH, 'r', encoding='utf-8') as f:
        raw_text = f.read()
    print(f"Successfully extracted {len(raw_text)} characters.")
except Exception as e:
    print(f"Error reading .txt file: {e}")
    exit()

if not raw_text.strip():
    print("Error: No text found in the file.")
    exit()

# --- 3. Split Text into Chunks ---
print("Splitting text into chunks...")
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000, # Max size of each chunk
    chunk_overlap=200, # How much chunks overlap
    length_function=len
)
chunks = text_splitter.split_text(raw_text)
print(f"Created {len(chunks)} text chunks.")

# --- 4. Load AI Model ---
print(f"Loading model: {MODEL_NAME}...")
model = SentenceTransformer(MODEL_NAME)

# --- 5. Create Embeddings ---
print("Creating embeddings for chunks... (This may take a while)")
embeddings = model.encode(chunks, show_progress_bar=True)
print(f"Embeddings created with shape: {embeddings.shape}")

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
    pickle.dump(chunks, f)

print("-" * 30)
print("✅ Ingestion Complete!")
print(f"Your new AI brain is saved in '{VECTOR_DB_PATH}' and '{DATA_STORE_PATH}'.")
print("-" * 30)
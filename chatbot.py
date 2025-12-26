import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
from dotenv import load_dotenv
load_dotenv()
import pickle
import faiss
# from sentence_transformers import SentenceTransformer # REMOVED
import numpy as np
from datetime import datetime
import threading
import requests
import json
import time

# --- Configuration ---
LOG_FILE = 'unanswered_log.txt'
# MODEL_NAME = 'all-MiniLM-L6-v2' # REMOVED
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
VECTOR_DB_PATH = os.path.join(BASE_DIR, "faiss_index")
DATA_STORE_PATH = os.path.join(BASE_DIR, "data_store.pkl")

# --- Gemini API Configuration ---
# --- IMPORTANT! PASTE YOUR KEY HERE ---
# API_KEY = "AIzaSyD1o1ACeFFm6eyYmpJOvPHiuIjiv3dDWJc" # OLD HARDCODED KEY
# API_KEY = os.environ.get("GEMINI_API_KEY")
API_KEY = os.getenv("GEMINI_API_KEY")
if not API_KEY or API_KEY == "Paste_Your_Gemini_API_Key_Here":
    print("WARNING: GEMINI_API_KEY not set in .env file.")
GEMINI_API_URL = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash-preview-09-2025:generateContent?key={API_KEY}"

# --- Cloud Embedding Function ---
def get_embedding(text):
    url = f"https://generativelanguage.googleapis.com/v1beta/models/text-embedding-004:embedContent?key={API_KEY}"
    payload = {
        "model": "models/text-embedding-004",
        "content": {
            "parts": [{"text": text}]
        }
    }
    headers = {'Content-Type': 'application/json'}
    try:
        response = requests.post(url, headers=headers, data=json.dumps(payload), timeout=10)
        if response.status_code == 200:
            return response.json()['embedding']['values']
        else:
            print(f"Error getting embedding: {response.text}")
            return None
    except Exception as e:
        print(f"Error: {e}")
        return None

# --- Load Model (Depreciated - Cloud used instead) ---
print("Configured for Cloud Embeddings.")
model = None # No local model

# --- Global variables for our 'brain' ---
index = None
chunks = []

def load_brain():
    """
    This function loads the FAISS index and the text chunks into memory.
    """
    global index, chunks
    if not os.path.exists(VECTOR_DB_PATH) or not os.path.exists(DATA_STORE_PATH):
        print("Error: FAISS index or data store not found. Run 'ingest.py' first.")
        return
    print("Loading FAISS index and data store...")
    try:
        index = faiss.read_index(VECTOR_DB_PATH)
        with open(DATA_STORE_PATH, 'rb') as f:
            chunks = pickle.load(f)
        print(f"✅ AI Brain (FAISS) loaded successfully with {len(chunks)} chunks.")
    except Exception as e:
        print(f"Error loading AI brain: {e}")

def log_unanswered_question(question):
    """
    Writes a question to our log file in a separate thread.
    """
    try:
        with open(LOG_FILE, 'a', encoding='utf-8') as f:
            timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            f.write(f"[{timestamp}] - {question}\n")
    except Exception as e:
        print(f"Error logging unanswered question: {e}")

# --- UPDATED: Generative Function (The "G" in RAG) ---
def get_generative_answer(context, question, chat_history, retries=3, backoff_factor=2):
    """
    Calls the Gemini API to generate a clean answer based on context AND chat history.
    Now includes safety settings and better error parsing.
    """
    print("DEBUG (Gemini): Generating clean answer with history...")
    system_prompt = (
        "You are a helpful and concise assistant, an expert in Design and Analysis of Algorithms (DAA) and Computer Science. "
        "Use the provided CONTEXT (which is a chunk from a textbook) to answer the user's new QUESTION. "
        "Use the CHAT HISTORY for context if the question is a follow-up (e.g., 'why?' or 'explain that')."
        "Answer *only* based on the context. Do not make up information. "
        "Be direct, clear, and explain concepts step-by-step if needed."
    )
    
    user_query = f"CONTEXT:\n---\n{context}\n---\n\nQUESTION: {question}"

    messages_for_api = chat_history + [
        {"role": "user", "parts": [{"text": user_query}]}
    ]

    payload = {
        "contents": messages_for_api,
        "systemInstruction": {
            "parts": [{"text": system_prompt}]
        },
        # --- THIS SECTION IS UPDATED ---
        "generationConfig": { 
            "temperature": 0.2, 
            "topK": 1, 
            "topP": 1, 
            "maxOutputTokens": 1024  # Increased from 256 to 1024
        },
        # --- END OF UPDATE ---
        "safetySettings": [
            {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_NONE"},
            {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_NONE"},
            {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_NONE"},
            {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_NONE"}
        ]
    }
    
    headers = {'Content-Type': 'application/json'}
    
    for i in range(retries):
        try:
            response = requests.post(GEMINI_API_URL, headers=headers, data=json.dumps(payload), timeout=20)
            
            if response.status_code == 200:
                result = response.json()
                
                try:
                    candidate = result.get('candidates', [])[0]
                    finish_reason = candidate.get('finishReason')
                    
                    if finish_reason and finish_reason != 'STOP':
                        print(f"DEBUG (Gemini): Generation stopped. Reason: {finish_reason}")
                        # If it was a MAX_TOKENS block, we still have *some* text
                        if finish_reason == "MAX_TOKENS" and candidate.get('content', {}).get('parts', [])[0].get('text'):
                             return candidate.get('content', {}).get('parts', [])[0].get('text') + " ... (answer shortened)"
                        return f"I found the info, but my summarization brain was blocked ({finish_reason}). Here is the raw text:\n\n{context}"

                    text = candidate.get('content', {}).get('parts', [])[0].get('text')

                    if text:
                        return text
                    else:
                        print(f"DEBUG (Gemini): No 'text' in candidate parts. {result}")
                        return f"I found the info, but had trouble parsing the summary. Here is the raw text:\n\n{context}"

                except (IndexError, KeyError, AttributeError, TypeError) as e:
                    print(f"DEBUG (Gemini): Error parsing response JSON: {e}. Full response: {result}")
                    return f"I found the info, but the summary was in a weird format. Here is the raw text:\n\n{context}"

            else:
                print(f"DEBUG (Gemini): API Error. Status: {response.status_code}, Response: {response.text}")
                if response.status_code == 400 and "API_KEY_INVALID" in response.text:
                     return "ERROR: The server's API key is invalid. Please check the backend configuration."

        except requests.exceptions.RequestException as e:
            print(f"DEBUG (Gemini): Request failed: {e}")
        
        if i < retries - 1:
            wait_time = backoff_factor ** i
            print(f"DEBUG (Gemini): Retrying in {wait_time}s...")
            time.sleep(wait_time)
    
    print("DEBUG (Gemini): All retries failed.")
    return f"I'm having trouble connecting to my summarization brain. Here is the raw text I found:\n\n{context}"

# --- Main Bot Response Function ---
def get_bot_response(user_question, chat_history):
    if index is None or not chunks:
        return "I'm sorry, my brain is not loaded. Please ask an admin to train me."

    try:
        # question_embedding = model.encode([user_question]) # OLD
        q_emb = get_embedding(user_question)
        if q_emb is None:
             return "I'm having trouble understanding (Embedding Error)."
        
        question_embedding = np.array([q_emb]).astype('float32')
        D, I = index.search(question_embedding, k=1)
        
        best_match_index = I[0][0]
        distance = D[0][0]

        print(f"DEBUG: User asked: '{user_question}'")
        print(f"DEBUG: Best match is chunk {best_match_index} with distance: {distance}")

        CONFIDENCE_THRESHOLD = 2.0 
        
        if distance < CONFIDENCE_THRESHOLD:
            context_chunk = chunks[best_match_index]
            generative_answer = get_generative_answer(context_chunk, user_question, chat_history)
            return generative_answer
        else:
            threading.Thread(target=log_unanswered_question, args=(user_question,)).start()
            return f"I'm sorry, I couldn't find a confident answer for that. (Best match distance: {distance:.2f} / Threshold: {CONFIDENCE_THRESHOLD})"

    except Exception as e:
        print(f"Error in get_bot_response: {e}") 
        return "An error occurred. Please try again."
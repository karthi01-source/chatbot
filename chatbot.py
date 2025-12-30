import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
os.environ['FAISS_OPT_LEVEL'] = 'generic'
from dotenv import load_dotenv
load_dotenv()
import pickle
import faiss
import numpy as np
from datetime import datetime
import threading
import requests
import json
import time

# --- Configuration ---
LOG_FILE = 'unanswered_log.txt'
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
VECTOR_DB_PATH = os.path.join(BASE_DIR, "faiss_index")
DATA_STORE_PATH = os.path.join(BASE_DIR, "data_store.pkl")

# --- Gemini API Configuration ---
API_KEY = os.getenv("GEMINI_API_KEY")
if not API_KEY or API_KEY == "Paste_Your_Gemini_API_Key_Here":
    print("WARNING: GEMINI_API_KEY not set in .env file.")

# Optimized list of models. We try the most likely to work/fastest first.
MODEL_CANDIDATES = [
    "gemini-2.0-flash-lite-preview-02-05", # Often has separate quota
    "gemini-flash-latest",                  # Stable 1.5 Flash
    "gemini-2.0-flash"                      # Standard 2.0 Flash
]

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
def get_generative_answer(context, question, chat_history, retries_per_model=1):
    """
    Calls the Gemini API to generate a clean answer based on context AND chat history.
    Iterates through MODEL_CANDIDATES. Fails gracefully if all are blocked.
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
        "generationConfig": { 
            "temperature": 0.2, 
            "topK": 1, 
            "topP": 1, 
            "maxOutputTokens": 1024 
        },
        "safetySettings": [
            {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_NONE"},
            {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_NONE"},
            {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_NONE"},
            {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_NONE"}
        ]
    }
    
    headers = {'Content-Type': 'application/json'}
    
    for model_name in MODEL_CANDIDATES:
        print(f"DEBUG: Trying model: {model_name}")
        api_url = f"https://generativelanguage.googleapis.com/v1beta/models/{model_name}:generateContent?key={API_KEY}"
        
        for i in range(retries_per_model + 1): # +1 for the initial try
            try:
                response = requests.post(api_url, headers=headers, data=json.dumps(payload), timeout=15)
                
                if response.status_code == 200:
                    result = response.json()
                    try:
                        candidate = result.get('candidates', [])[0]
                        finish_reason = candidate.get('finishReason')
                        
                        if finish_reason and finish_reason != 'STOP':
                            if finish_reason == "MAX_TOKENS" and candidate.get('content', {}).get('parts', [])[0].get('text'):
                                 return candidate.get('content', {}).get('parts', [])[0].get('text') + " ... (answer shortened)"
                            # If blocked, try next model? Or just return raw text?
                            # Usually safety block.
                            print(f"DEBUG: Blocked by safety/other: {finish_reason}")
                            break 

                        text = candidate.get('content', {}).get('parts', [])[0].get('text')
                        if text:
                            return text
                        else:
                            print(f"DEBUG (Gemini): No 'text' in candidate parts. {result}")
                            break

                    except (IndexError, KeyError, AttributeError, TypeError) as e:
                        print(f"DEBUG (Gemini): Error parsing response JSON: {e}")
                        break

                elif response.status_code == 429:
                    print(f"DEBUG (Gemini): Rate limit hit for {model_name}. Retrying...")
                    time.sleep(1) # Short wait
                elif response.status_code == 404:
                    print(f"DEBUG (Gemini): Model {model_name} not found. Skipping.")
                    break 
                else:
                    print(f"DEBUG (Gemini): API Error {response.status_code} for {model_name}")
                    break

            except requests.exceptions.RequestException as e:
                print(f"DEBUG (Gemini): Request failed: {e}")
                time.sleep(1)
        
    
    print("DEBUG (Gemini): All models failed or were blocked.")
    # Graceful Fallback: Just show the text nicely.
    return f"**Note:** I'm currently experiencing high traffic on my summarization engine. Here is the relevant information directly from the handbook:\n\n{context}"

# --- Main Bot Response Function ---
def get_bot_response(user_question, chat_history):
    global index, chunks
    if index is None or not chunks:
        load_brain()
        
    if index is None or not chunks:
        return "I'm sorry, my brain is not loaded. Please ask an admin to train me."

    try:
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
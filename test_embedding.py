import requests
import json
import os

# Copying the key logic from chatbot.py
API_KEY = os.environ.get("GEMINI_API_KEY", "AIzaSyD1o1ACeFFm6eyYmpJOvPHiuIjiv3dDWJc")
print(f"Using API Key: {API_KEY[:5]}...{API_KEY[-5:]}")

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
        print(f"Sending request to {url}...")
        response = requests.post(url, headers=headers, data=json.dumps(payload), timeout=10)
        print(f"Response Status Code: {response.status_code}")
        if response.status_code == 200:
            print("Success!")
            return response.json()['embedding']['values']
        else:
            print(f"Error getting embedding: {response.text}")
            return None
    except Exception as e:
        print(f"Error: {e}")
        return None

if __name__ == "__main__":
    emb = get_embedding("Hello world")
    if emb:
        print(f"Embedding received, length: {len(emb)}")
    else:
        print("Failed to get embedding.")

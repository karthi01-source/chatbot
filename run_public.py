import os
import sys
import threading
from pyngrok import ngrok
from flask import Flask

# Add current directory to path
sys.path.append(os.path.abspath(os.path.dirname(__file__)))

# Import the app and chatbot
import chatbot
from app import app

def run_app():
    # Load the brain
    print("Loading AI Brain...")
    chatbot.load_brain()
    
    # Start ngrok
    print("Starting ngrok tunnel...")
    # Open a HTTP tunnel on the default port 5000
    public_url = ngrok.connect(5000).public_url
    print(f" * Public URL: {public_url}")
    print(f" * Local URL: http://127.0.0.1:5000")

    # Run the app
    app.run(port=5000, use_reloader=False)

if __name__ == '__main__':
    run_app()

from flask import Flask, request, jsonify, render_template, redirect, url_for
import chatbot  # Import our chatbot logic
from datetime import datetime
import threading
import os
import re

app = Flask(__name__)
UNANSWERED_LOG = 'unanswered_log.txt'
FEEDBACK_LOG = 'feedback_log.txt'

# --- Feedback Logger ---
def log_feedback(question, answer, feedback_type):
    try:
        with open(FEEDBACK_LOG, 'a', encoding='utf-8') as f:
            timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            f.write(f"[{timestamp}] - TYPE: {feedback_type}\n")
            f.write(f"  Q: {question}\n")
            f.write(f"  A: {answer}\n")
            f.write("-" * 20 + "\n")
    except Exception as e:
        print(f"Error logging feedback: {e}")

# --- Admin Log Parsers ---
def parse_unanswered_logs():
    logs = []
    if not os.path.exists(UNANSWERED_LOG): return logs
    with open(UNANSWERED_LOG, 'r', encoding='utf-8') as f:
        for line in f:
            match = re.match(r"\[(.*?)\] - (.*)", line)
            if match: logs.append({'timestamp': match.group(1), 'question': match.group(2)})
    return logs

def parse_feedback_logs():
    logs = []
    if not os.path.exists(FEEDBACK_LOG): return logs
    with open(FEEDBACK_LOG, 'r', encoding='utf-8') as f:
        content = f.read().split('-' * 20)
        for entry in content:
            if not entry.strip(): continue
            log = {}
            ts_match = re.search(r"\[(.*?)\] - TYPE: (up|down)", entry)
            q_match = re.search(r"Q: (.*)", entry)
            a_match = re.search(r"A: (.*)", entry, re.DOTALL)
            if ts_match and q_match and a_match:
                log.update({
                    'timestamp': ts_match.group(1), 'type': ts_match.group(2),
                    'question': q_match.group(1).strip(), 'answer': a_match.group(1).strip()
                })
                logs.append(log)
    return logs

# --- Routes ---

@app.route('/')
def home():
    return render_template('index.html')

# --- THIS IS THE UPDATED /ask ROUTE ---
@app.route('/ask', methods=['POST'])
def ask():
    data = request.get_json()
    user_question = data.get('question')
    chat_history = data.get('history', []) # Get history, default to empty list

    if not user_question:
        return jsonify({'answer': 'Invalid request. No question provided.'}), 400
    
    # Pass both to the chatbot
    bot_answer = chatbot.get_bot_response(user_question, chat_history)
    
    return jsonify({'answer': bot_answer})
# --- END OF UPDATE ---

@app.route('/feedback', methods=['POST'])
def feedback():
    data = request.get_json()
    question = data.get('question')
    answer = data.get('answer')
    feedback_type = data.get('feedback_type')
    if not all([question, answer, feedback_type]):
        return jsonify({'status': 'error', 'message': 'Missing data'}), 400
    threading.Thread(target=log_feedback, args=(question, answer, feedback_type)).start()
    return jsonify({'status': 'success', 'message': 'Feedback received'})

@app.route('/admin')
def admin():
    unanswered = parse_unanswered_logs()
    feedback = parse_feedback_logs()
    unanswered.reverse()
    feedback.reverse()
    return render_template('admin.html', unanswered_logs=unanswered, feedback_logs=feedback)

@app.route('/clear_logs')
def clear_logs():
    try:
        if os.path.exists(UNANSWERED_LOG): os.remove(UNANSWERED_LOG)
        if os.path.exists(FEEDBACK_LOG): os.remove(FEEDBACK_LOG)
    except Exception as e:
        print(f"Error clearing logs: {e}")
    return redirect(url_for('admin'))

# --- Run the App ---
if __name__ == '__main__':
    print("Starting Flask server...")
    chatbot.load_brain()  # Load the FAISS brain
    app.run(debug=True)
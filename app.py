import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['FAISS_OPT_LEVEL'] = 'generic'
from flask import Flask, request, jsonify, render_template, redirect, url_for
import chatbot  # Import our chatbot logic
from datetime import datetime
import threading
import re
from werkzeug.utils import secure_filename
import ingest

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'data_uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB limit
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
UNANSWERED_LOG = 'unanswered_log.txt'
FEEDBACK_LOG = 'feedback_log.txt'

# --- Ingestion Status Tracking ---
INGESTION_STATUS = {
    'status': 'Idle',
    'last_run': None,
    'error': None
}

# --- Load Brain (Critical for Gunicorn) ---
# Removed top-level load to prevent hang

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
    
    # List uploaded files
    uploaded_files = []
    if os.path.exists(app.config['UPLOAD_FOLDER']):
        supported_extensions = ('.txt', '.pdf')
        uploaded_files = [f for f in os.listdir(app.config['UPLOAD_FOLDER']) 
                          if f.lower().endswith(supported_extensions)]
        
    return render_template('admin.html', 
                           unanswered_logs=unanswered, 
                           feedback_logs=feedback, 
                           uploaded_files=uploaded_files)

@app.route('/clear_logs')
def clear_logs():
    try:
        if os.path.exists(UNANSWERED_LOG): os.remove(UNANSWERED_LOG)
        if os.path.exists(FEEDBACK_LOG): os.remove(FEEDBACK_LOG)
    except Exception as e:
        print(f"Error clearing logs: {e}")
    return redirect(url_for('admin'))

@app.route('/upload_doc', methods=['POST'])
def upload_doc():
    if 'file' not in request.files:
        return redirect(url_for('admin', status='No file part'))
    
    files = request.files.getlist('file')
    if not files or files[0].filename == '':
        return redirect(url_for('admin', status='No selected files'))
    
    saved_count = 0
    for file in files:
        if file:
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)
            saved_count += 1
    
    if saved_count > 0:
        # Trigger Background Ingestion
        def run_ingestion():
            global INGESTION_STATUS
            INGESTION_STATUS['status'] = 'Processing'
            INGESTION_STATUS['last_run'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            INGESTION_STATUS['error'] = None
            try:
                success = ingest.rebuild_brain(app.config['UPLOAD_FOLDER'])
                if success:
                    chatbot.load_brain()
                    INGESTION_STATUS['status'] = 'Success'
                else:
                    INGESTION_STATUS['status'] = 'Failed'
                    INGESTION_STATUS['error'] = 'Ingestion produced no chunks (check file types/content)'
            except Exception as e:
                INGESTION_STATUS['status'] = 'Error'
                INGESTION_STATUS['error'] = str(e)

        threading.Thread(target=run_ingestion).start()
        return redirect(url_for('admin', status=f'Uploaded {saved_count} files. Training started in background...'))
    
    return redirect(url_for('admin', status='No files were saved.'))

@app.route('/ingest_status')
def ingest_status():
    return jsonify(INGESTION_STATUS)

@app.route('/delete_doc/<filename>')
def delete_doc(filename):
    filename = secure_filename(filename)
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    
    if os.path.exists(filepath):
        try:
            os.remove(filepath)
            # Rebuild brain after deletion
            ingest.rebuild_brain(app.config['UPLOAD_FOLDER'])
            chatbot.load_brain()
            return redirect(url_for('admin', status=f'File {filename} deleted and brain updated.'))
        except Exception as e:
            return redirect(url_for('admin', status=f'Error deleting file: {e}'))
    else:
        return redirect(url_for('admin', status='File not found.'))

@app.route('/delete_all_docs')
def delete_all_docs():
    try:
        if os.path.exists(app.config['UPLOAD_FOLDER']):
            for f in os.listdir(app.config['UPLOAD_FOLDER']):
                if f.lower().endswith(('.txt', '.pdf')):
                    os.remove(os.path.join(app.config['UPLOAD_FOLDER'], f))
        
        # Rebuild brain (which will now clear it)
        ingest.rebuild_brain(app.config['UPLOAD_FOLDER'])
        
        # Update memory
        chatbot.index = None
        chatbot.chunks = []
        
        return redirect(url_for('admin', status='All documents deleted and brain cleared.'))
    except Exception as e:
        return redirect(url_for('admin', status=f'Error clearing documents: {e}'))


# --- Run the App ---
if __name__ == '__main__':
    print("\n!!! STARTING FLASK SERVER ON http://127.0.0.1:5000 !!!\n")
    chatbot.load_brain()  # Load the FAISS brain
    app.run(debug=True, port=5000)
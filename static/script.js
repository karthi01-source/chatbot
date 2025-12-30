// --- Get Elements ---
const chatBox = document.getElementById('chat-box');
const userInput = document.getElementById('user-input');
const sendBtn = document.getElementById('send-btn');
const themeToggleBtn = document.getElementById('theme-toggle');
const clearBtn = document.getElementById('clear-btn');
const micBtn = document.getElementById('mic-btn');

// --- Speech API Setup ---
const SpeechRecognition = window.SpeechRecognition || window.webkitSpeechRecognition;
const recognition = SpeechRecognition ? new SpeechRecognition() : null;
const speechSynthesis = window.speechSynthesis;

if (recognition) {
    recognition.continuous = false;
    recognition.lang = 'en-US';
    recognition.interimResults = false;
    recognition.maxAlternatives = 1;

    micBtn.addEventListener('click', () => {
        try {
            recognition.start();
            userInput.placeholder = "Listening...";
        } catch (e) { console.error("Speech recognition already active.", e); }
    });
    recognition.onresult = (event) => {
        userInput.value = event.results[0][0].transcript;
        sendMessage();
    };
    recognition.onend = () => { userInput.placeholder = "Type your message..."; };
    recognition.onerror = (event) => { console.error("Speech recognition error:", event.error); userInput.placeholder = "Type your message..."; };
} else {
    micBtn.style.display = 'none';
    console.warn("SpeechRecognition API not supported.");
}

// --- Event Listeners ---
sendBtn.addEventListener('click', sendMessage);
userInput.addEventListener('keypress', (e) => {
    if (e.key === 'Enter') {
        e.preventDefault();
        sendMessage();
    }
});

themeToggleBtn.addEventListener('click', () => {
    document.body.classList.toggle('dark-mode');
    localStorage.setItem('theme', document.body.classList.contains('dark-mode') ? 'dark' : 'light');
});

clearBtn.addEventListener('click', () => {
    chatBox.innerHTML = `
        <div class="message bot-message" data-role="model">
            <p>Hello! I'm the college AI assistant. How can I help you today?</p>
            ${createSpeakButton("Hello! I'm the college AI assistant. How can I help you today?")}
        </div>
    `;
    if (speechSynthesis.speaking) speechSynthesis.cancel();
    // Re-run MathJax on the welcome message after clearing
    if (window.MathJax) {
        MathJax.typesetPromise([chatBox.querySelector('.bot-message')]).catch((err) => console.log('MathJax error:', err));
    }
});

// --- On Page Load ---
document.addEventListener('DOMContentLoaded', () => {
    const savedTheme = localStorage.getItem('theme');
    if (savedTheme === 'dark') document.body.classList.add('dark-mode');
    // Add speak button and typeset the initial message
    const firstBotMsg = chatBox.querySelector('.bot-message');
    if (firstBotMsg) {
        const p = firstBotMsg.querySelector('p');
        firstBotMsg.innerHTML += createSpeakButton(p.textContent);
        if (window.MathJax) {
            MathJax.typesetPromise([firstBotMsg]).catch((err) => console.log('MathJax error:', err));
        }
    }
});

// --- Helper to get chat history ---
function getChatHistory() {
    const messages = chatBox.querySelectorAll('.message');
    const history = [];
    // Get the last 4 messages (or fewer if not available)
    const recentMessages = Array.from(messages).slice(-4);

    recentMessages.forEach(msg => {
        const role = msg.dataset.role; // We'll add this dataset attribute
        const text = msg.querySelector('p') ? msg.querySelector('p').textContent : msg.querySelector('div').textContent; // Handle p or div
        if (role && text) {
            history.push({ "role": role, "parts": [{ "text": text }] });
        }
    });
    return history;
}

// --- Core Send Function ---
function sendMessage() {
    if (speechSynthesis.speaking) speechSynthesis.cancel();

    const userMessageText = userInput.value.trim();
    if (userMessageText === '') return;

    // Get history *before* adding the new message
    const chatHistory = getChatHistory();

    appendMessage(userMessageText, 'user');
    userInput.value = '';
    showTypingIndicator();

    // Send the new question AND the history
    fetch('/ask', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
            'question': userMessageText,
            'history': chatHistory  // Send the history
        }),
    })
        .then(response => response.json())
        .then(data => {
            hideTypingIndicator();
            appendMessage(data.answer, 'bot', userMessageText);
        })
        .catch((error) => {
            console.error('Error:', error);
            hideTypingIndicator();
            appendMessage('Sorry, something went wrong. Please try again.', 'bot');
        });
}

// --- appendMessage function (with MathJax) ---
function appendMessage(message, sender, originalQuestion = null) {
    const messageDiv = document.createElement('div');
    messageDiv.classList.add('message', `${sender}-message`);
    messageDiv.dataset.role = (sender === 'user') ? 'user' : 'model';

    const messageP = document.createElement('div'); // Changed to div for HTML content
    // Render Markdown
    if (sender === 'bot') {
        messageP.innerHTML = marked.parse(message);
    } else {
        messageP.textContent = message;
    }
    messageDiv.appendChild(messageP);

    // Highlight Code Blocks
    if (sender === 'bot') {
        messageDiv.querySelectorAll('pre code').forEach((block) => {
            hljs.highlightElement(block);
        });
    }

    if (sender === 'bot') {
        messageDiv.innerHTML += createSpeakButton(message);
        if (originalQuestion && !message.startsWith("I'm sorry") && !message.startsWith("ERROR:")) {
            const feedbackDiv = document.createElement('div');
            feedbackDiv.classList.add('feedback-container');
            const thumbUpBtn = createFeedbackButton('👍', () => {
                sendFeedback(originalQuestion, message, 'up');
                thumbUpBtn.disabled = true;
                thumbDownBtn.disabled = true;
            });
            const thumbDownBtn = createFeedbackButton('👎', () => {
                sendFeedback(originalQuestion, message, 'down');
                thumbUpBtn.disabled = true;
                thumbDownBtn.disabled = true;
            });
            feedbackDiv.appendChild(thumbUpBtn);
            feedbackDiv.appendChild(thumbDownBtn);
            messageDiv.appendChild(feedbackDiv);
        }
    }
    chatBox.appendChild(messageDiv);

    // --- IMPORTANT: Tell MathJax to render the new message ---
    if (window.MathJax) {
        MathJax.typesetPromise([messageDiv]).catch((err) => console.log('MathJax typeset error:', err));
    }
    // --- END ---

    chatBox.scrollTop = chatBox.scrollHeight;
}

// --- Text-to-Speech Functions ---
function createSpeakButton(text) {
    if (!speechSynthesis) return '';
    // Use encodeURIComponent to safely pass text
    return `<button class="speak-btn" onclick="speakText(this, decodeURIComponent('${encodeURIComponent(text)}'))">
        <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="currentColor"><path d="M3 9v6h4l5 5V4L7 9H3zm13.5 3c0-1.77-1.02-3.29-2.5-4.03v8.05c1.48-.73 2.5-2.25 2.5-4.02zM14 3.23v2.06c2.89.86 5 3.54 5 6.71s-2.11 5.85-5 6.71v2.06c4.01-.91 7-4.49 7-8.77s-2.99-7.86-7-8.77z"/></svg>
        </button>`;
}

function speakText(button, text) {
    if (speechSynthesis.speaking) speechSynthesis.cancel();
    const utterance = new SpeechSynthesisUtterance(text);
    utterance.lang = 'en-US';
    utterance.onend = () => button.classList.remove('speaking');
    speechSynthesis.speak(utterance);
    button.classList.add('speaking');
}

// --- Helper & Feedback Functions ---
function createFeedbackButton(text, onClick) {
    const btn = document.createElement('button');
    btn.classList.add('feedback-btn');
    btn.innerHTML = text;
    btn.onclick = onClick;
    return btn;
}

function sendFeedback(question, answer, feedbackType) {
    fetch('/feedback', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ 'question': question, 'answer': answer, 'feedback_type': feedbackType }),
    });
}

// --- Typing Indicator Functions ---
function showTypingIndicator() {
    if (document.getElementById('typing-indicator')) return;
    const typingDiv = document.createElement('div');
    typingDiv.id = 'typing-indicator';
    typingDiv.classList.add('typing-indicator');
    typingDiv.innerHTML = '<span></span><span></span><span></span>';
    chatBox.appendChild(typingDiv);
    chatBox.scrollTop = chatBox.scrollHeight;
}

function hideTypingIndicator() {
    const typingDiv = document.getElementById('typing-indicator');
    if (typingDiv) typingDiv.remove();
}
import os
import uuid
import json
import requests
import logging
import re
from flask import Flask, render_template, request, jsonify, send_from_directory
from PyPDF2 import PdfReader

UPLOAD_FOLDER = "uploads"
STATE_FILE = "chatbot_state.json"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

app = Flask(__name__, static_folder="static", template_folder="templates")
app.secret_key = "super-secret-key"
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

# Configure logging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')

# Helper functions to persist state in a file
def save_state(file_id, state):
    with open(STATE_FILE, "w") as f:
        json.dump({"file_id": file_id, **state}, f)

def load_state():
    if not os.path.exists(STATE_FILE):
        return None, {}
    with open(STATE_FILE, "r") as f:
        state = json.load(f)
    return state.get("file_id"), {k: v for k, v in state.items() if k != "file_id"}

# Function to extract text from a PDF file
def extract_text_from_pdf(pdf_path):
    reader = PdfReader(pdf_path)
    text = ""
    for page in reader.pages:
        text += page.extract_text() + "\n"
    return text

# Function to clean text before processing
def clean_text(text):
    # Remove special characters and extra whitespace
    text = re.sub(r"[^a-zA-Z0-9\s]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    # Convert text to lowercase
    text = text.lower()
    return text

# Correctly handle the HuggingFace API response
def summarize_with_huggingface(prompt):
    api_key = os.environ.get("HF_TOKEN")
    if not api_key:
        raise Exception("HuggingFace API key (HF_TOKEN) is not set. Please set it as an environment variable.")

    # Clean and truncate the input text
    max_input_length = 1024
    cleaned_prompt = clean_text(prompt)
    truncated_prompt = cleaned_prompt[:max_input_length]

    url = "https://api-inference.huggingface.co/models/facebook/bart-large-cnn"
    headers = {"Authorization": f"Bearer {api_key}"}
    payload = {"inputs": truncated_prompt}

    logging.debug(f"API Key: {api_key}")
    logging.debug(f"Payload: {payload}")
    response = requests.post(url, headers=headers, json=payload)
    logging.debug(f"Response Status Code: {response.status_code}")
    logging.debug(f"Response Text: {response.text}")

    if response.status_code == 200:
        # Ensure the response is correctly parsed
        result = response.json()
        if isinstance(result, list) and len(result) > 0:
            return result[0].get("summary_text", "No summary generated.")
        else:
            raise Exception("Unexpected response format from HuggingFace API.")
    elif response.status_code == 401:
        raise Exception("Invalid HuggingFace API key. Please verify your HF_TOKEN.")
    else:
        raise Exception(f"HuggingFace API error: {response.status_code}, {response.text}")

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/upload", methods=["POST"])
def upload():
    if 'file' not in request.files:
        return jsonify({"error": "No file part"}), 400
    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400

    ext = os.path.splitext(file.filename)[1]
    file_id = str(uuid.uuid4())
    filename = f"{file_id}{ext}"
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(filepath)

    # Extract text from the uploaded PDF
    pdf_text = extract_text_from_pdf(filepath)

    # Save the extracted text along with the file ID
    save_state(file_id, {"text": pdf_text, "chat_history": []})
    return jsonify({"success": True})

@app.route("/chat", methods=["POST"])
def chat():
    data = request.get_json()
    user_input = data.get("message", "")
    file_id, state = load_state()

    if not file_id:
        return jsonify({"answer": "No document has been uploaded yet."})

    pdf_text = state.get("text", "")
    chat_history = state.get("chat_history", [])

    if "summary" in user_input.lower():
        # Generate a summary using the HuggingFace API
        try:
            answer = summarize_with_huggingface(user_input + "\n" + pdf_text)
        except Exception as e:
            answer = f"Error generating summary: {str(e)}"
    else:
        # Generate a response based on the full extracted text
        answer = f"Response for question: '{user_input}' based on extracted text: {pdf_text}"

    chat_history.append((user_input, answer))
    save_state(file_id, {"text": pdf_text, "chat_history": chat_history})
    return jsonify({"answer": answer})

@app.route("/uploads/<filename>")
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

if __name__ == "__main__":
    app.run(debug=False, use_reloader=False, host="0.0.0.0")
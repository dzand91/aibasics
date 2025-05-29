import os
import uuid
import json
from flask import Flask, render_template, request, jsonify, send_from_directory
from worker import process_document, get_qa_chain

UPLOAD_FOLDER = "uploads"
VECTORDB_FOLDER = "vectordbs"
STATE_FILE = "chatbot_state.json"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(VECTORDB_FOLDER, exist_ok=True)

app = Flask(__name__, static_folder="static", template_folder="templates")
app.secret_key = "super-secret-key"
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

# Helper functions to persist state across processes
def save_state(file_id, chat_history):
    with open(STATE_FILE, "w") as f:
        json.dump({"last_file_id": file_id, "chat_history": chat_history}, f)

def load_state():
    if not os.path.exists(STATE_FILE):
        return None, []
    with open(STATE_FILE, "r") as f:
        data = json.load(f)
        return data.get("last_file_id"), data.get("chat_history", [])

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
    vectordb_dir = os.path.join(VECTORDB_FOLDER, file_id)
    process_document(filepath, vectordb_dir)

    save_state(file_id, [])  # Persist file id and empty chat history
    return jsonify({"success": True})

@app.route("/chat", methods=["POST"])
def chat():
    data = request.get_json()
    user_input = data.get("message", "")
    file_id, chat_history = load_state()

    if not file_id:
        return jsonify({"answer": "No document has been uploaded yet."})

    vectordb_dir = os.path.join(VECTORDB_FOLDER, file_id)
    if not os.path.exists(vectordb_dir):
        return jsonify({"answer": "Document processing failed or expired."})

    # Optionally re-process document for safety
    pdf_path = None
    uploads = os.listdir(UPLOAD_FOLDER)
    for fname in uploads:
        if fname.startswith(file_id):
            pdf_path = os.path.join(UPLOAD_FOLDER, fname)
            break
    if pdf_path and os.path.exists(pdf_path):
        process_document(pdf_path, vectordb_dir)

    qa_chain = get_qa_chain(vectordb_dir)
    output = qa_chain.invoke({"question": user_input, "chat_history": chat_history})
    answer = output["result"]
    chat_history.append((user_input, answer))
    save_state(file_id, chat_history)  # Save updated chat history
    return jsonify({"answer": answer})

@app.route("/uploads/<filename>")
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

if __name__ == "__main__":
    app.run(debug=False, use_reloader=False, host="0.0.0.0")
import os
import sys
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
from dotenv import load_dotenv

# --- Fix Module Import Path ---
root_path = os.path.join(os.path.dirname(__file__), '..')
if root_path not in sys.path:
    sys.path.append(root_path)

# Import the RAG chain logic
from src.ingestion.chain.rag_chain import run_chain

load_dotenv(os.path.join(root_path, ".env"))

app = Flask(__name__, static_folder="../frontend", static_url_path="")
CORS(app, resources={r"/api/*": {"origins": "*"}}) # Allow cross-origin requests for the API

# --- Load HR Data for Login ---
import os
HR_DATA_PATH = os.path.join(os.path.dirname(__file__), "..", "Data", "hr", "hr_data.csv")
import pandas as pd
try:
    hr_df = pd.read_csv(HR_DATA_PATH)
except:
    hr_df = None

# --- Routes for Serving HTML ---

@app.route("/")
def index():
    return send_from_directory(app.static_folder, "bot.html")

@app.route("/dashboard")
def dashboard():
    return send_from_directory(app.static_folder, "bot2.html")

@app.route("/chat")
def chat():
    return send_from_directory(app.static_folder, "bot3.html")

# --- API Endpoints ---

@app.route("/api/login", methods=["POST"])
def api_login():
    data = request.json
    login_id = data.get("login_id")
    password = data.get("password")

    if hr_df is None:
        return jsonify({"error": "HR Database not found."}), 500

    # Search for login_id
    user_row = hr_df[hr_df['login_id'].str.lower() == login_id.lower()]
    
    if user_row.empty:
        return jsonify({"error": "Invalid Employee ID."}), 401
    
    user_data = user_row.iloc[0]
    
    # Password check (last 4 chars of ID)
    if password != login_id[-4:]:
        return jsonify({"error": "Incorrect password."}), 401

    # Map department to internal role
    dept = user_data['department']
    role_map = {
        'HR': 'hr_team',
        'Finance': 'finance_team',
        'Technology': 'engineering_team',
        'Marketing': 'marketing_team',
        'Sales': 'marketing_team', # fallback
        'Data': 'engineering_team', # fallback
        'Design': 'engineering_team', # fallback
        'Risk': 'finance_team', # fallback
        'Compliance': 'hr_team', # fallback
    }

    role = role_map.get(dept, "employee")
    
    return jsonify({
        "success": True,
        "name": user_data['full_name'],
        "role": role,
        "specific_role": user_data['role'],
        "department": dept,
        "email": user_data['email'],
        "employee_id": user_data['employee_id'],
        "location": user_data['location'],
        "date_of_joining": user_data['date_of_joining']
    })

@app.route("/api/chat", methods=["POST"])
def api_chat():
    data = request.json
    question = data.get("question")
    role = data.get("role", "employee")
    history = data.get("history", [])
    
    # Extract user personalized info sent from frontend
    user_name = data.get("user_name", "User")
    user_department = data.get("user_department", "Unknown Department")
    specific_role = data.get("specific_role", "Employee")
    
    user_context = {
        "name": user_name,
        "department": user_department,
        "role": specific_role
    }
    
    # Extract role for retriever filtering
    role = data.get("role", "employee")
    
    print(f"DEBUG: Request from {user_name} ({role}) - Quest: {question[:50]}...")

    if not question:
        return jsonify({"error": "No question provided"}), 400

    try:
        from langchain_core.messages import HumanMessage, AIMessage
        formatted_history = []
        for i in range(0, len(history), 2):
            if i + 1 < len(history):
                formatted_history.append(HumanMessage(content=history[i]))
                formatted_history.append(AIMessage(content=history[i+1]))

        result = run_chain(question, formatted_history, role=role, user_context=user_context)
        
        return jsonify({
            "answer": result["answer"],
            "sources": [doc.metadata.get("source", "unknown") for doc in result.get("source_documents", [])]
        })
    except Exception as e:
        print(f"Error in /api/chat: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False, threaded=True)

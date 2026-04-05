# AtliQ Insight Hub - Smart Chat Assistant 🚀

| **Project Hub** | **Live Deployment** |
| :--- | :--- |
| 🔗 [GitHub Repository](https://github.com/Ashu4495/AtliQ-Insight-Hub) | ⚡ [Production App](https://chatbot-app-265378659018.us-central1.run.app) |

---

AtliQ Insight Hub is a professional, RAG-powered (Retrieval-Augmented Generation) chatbot designed for enterprise intelligence. It allows employees to securely query internal company data across Departments such as **HR**, **Finance**, and **Engineering**.

---

## 📑 Table of Contents
1.  [Quick Start: How to Run](#-quick-start-how-to-run)
2.  [Key Features](#-key-features)
3.  [Technology Stack](#-technology-stack)
4.  [Deployment Guide (Google Cloud Run)](#-deployment-guide-google-cloud-run)
5.  [Environment Variables](#-environment-variables)
6.  [Project Structure](#-project-structure)
7.  [Privacy & Security](#-privacy--security)

---

## ⚡ Quick Start: How to Run

Follow these steps to run the application on your local machine:

### 1. Environment Setup
Clone the repository and create a virtual environment:
```bash
# Create and activate virtual environment
python -m venv venv
./venv/Scripts/activate  # Windows
# source venv/bin/activate # Linux/Mac
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Add Your Secrets
Create a `.env` file in the root directory:
```env
GROQ_API_KEY=your_groq_api_key_here
LANGCHAIN_API_KEY=your_langchain_api_key_here
# Optional: Add tracing
LANGCHAIN_TRACING_V2=true
```

### 4. Launch the App
```bash
python src/app.py
```
Open your browser and visit: **[http://127.0.0.1:5000](http://127.0.0.1:5000)**

---

## 🌟 Key Features

- **Strategic Dashboard:** A premium, interactive home screen with quick-access cards for "Company Policy," "Payroll," and "Tech Docs."
- **RAG-Powered Intelligence:** Uses LangChain and ChromaDB to provide accurate, context-aware answers from internal PDF/Text/CSV data.
- **Unified Global Input:** A modern, floating search bar that manages the transition between dashboard and active chat streams.
- **Knowledge Health Monitoring:** Real-time visualization of data synchronization status and indexing counts.
- **Professional Persona:** High-fidelity UI with glassmorphism modals, tailored typography (Inter/Manrope), and premium CSS animations.

---

## 🛠️ Technology Stack

- **Frontend:** HTML5, TailwindCSS, Vanilla JavaScript, Material Symbols.
- **Backend:** Flask (Python), CORS enabled.
- **LLM Engine:** LangChain + Groq API (Llama3-70b).
- **Vector Database:** ChromaDB (Local state).
- **Embeddings:** HuggingFace Sentence Transformers (`all-mpnet-base-v2`).
- **Data Guard:** Presidio PII Scanning.

---

## ☁️ Deployment Guide (Google Cloud Run)

### 1. Fix 'gcloud' Command Not Found
If you see an error like `gcloud : The term 'gcloud' is not recognized`, you must install the **Google Cloud SDK**:
- **Download Link:** [Google Cloud CLI Installer for Windows](https://dl.google.com/dl/cloudsdk/channels/rapid/GoogleCloudSDKInstaller.exe).
- **Install & Restart:** Ensure "Add to PATH" is checked, then restart your terminal.

### 2. Login & Deploy
```bash
# Login
gcloud auth login

# Deploy with Source
gcloud run deploy atliq-chatbot --source . --region us-central1 --allow-unauthenticated
```
*After deployment, copy the generated URL and paste it into the "Live Application" link at the top of this file.*

---

## 📁 Project Structure

```text
├── Data/                 # Source documents for RAG (HR, Finance, Tech)
├── chroma_db/            # Persistent Vector Storage
├── frontend/             # Desktop/Mobile UI Assets
│   ├── bot.html          # Login Page
│   └── bot3.html         # Premium Smart Chat (Latest)
├── src/                  # Backend Logic
│   ├── app.py            # Flask API Entry Point
│   └── ingestion/        # LangChain & ChromaDB Pipeline
├── Dockerfile            # Container Configuration
└── requirements.txt      # Dependency List
```

---

## 🔒 Privacy & Security
- **PII Guard:** Every query and AI response is scanned for Emails, SSNs, and Credit Cards.
- **Role-Based Filters:** Backend logic ensures users only access data relevant to their department.

---
*Created and maintained by the AtliQ Insight Hub Team.*

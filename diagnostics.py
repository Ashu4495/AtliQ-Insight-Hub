import os
import sys
from dotenv import load_dotenv

# Add project root to sys.path
root_path = os.path.abspath(os.path.dirname(__file__))
if root_path not in sys.path:
    sys.path.append(root_path)

load_dotenv()

def run_diagnostics():
    print("=== Chatbot Diagnostics ===")
    
    # 1. Check Environment Variables
    groq_api_key = os.getenv("GROQ_API_KEY")
    if not groq_api_key:
        print("[FAIL] GROQ_API_KEY is not set in .env or environment.")
    else:
        print("[OK] GROQ_API_KEY is found.")

    # 2. Check Vector Store
    chroma_path = os.path.join(root_path, "chroma_db")
    if not os.path.exists(chroma_path):
        print(f"[FAIL] ChromaDB directory NOT found at {chroma_path}")
    else:
        print(f"[OK] ChromaDB directory found at {chroma_path}")
        try:
            from src.ingestion.vectorstore.chroma_store import load_vectorstore
            vs = load_vectorstore()
            print(f"[OK] Vector store loaded successfully. Collection count: {vs._collection.count()}")
        except Exception as e:
            print(f"[FAIL] Failed to load vector store: {e}")

    # 3. Check LLM Connectivity
    try:
        from src.ingestion.chain.rag_chain import _get_llm
        llm = _get_llm()
        print("[OK] LLM (ChatGroq) initialized.")
        
        from langchain_core.messages import HumanMessage
        print("Testing Groq API connectivity...")
        response = llm.invoke([HumanMessage(content="Hello, are you there?")])
        print(f"[OK] Groq API responded: {response.content[:50]}...")
    except Exception as e:
        print(f"[FAIL] Groq API / LLM error: {e}")

    # 4. Check RAG Chain
    try:
        from src.ingestion.chain.rag_chain import run_chain
        print("Testing RAG Chain...")
        result = run_chain("What is the attendance policy?", [], role="employee")
        if "answer" in result:
            print(f"[OK] RAG Chain produced an answer: {result['answer'][:100]}...")
        else:
            print("[FAIL] RAG Chain returned a result without an 'answer' key.")
    except Exception as e:
        print(f"[FAIL] RAG Chain error: {e}")

if __name__ == "__main__":
    run_diagnostics()

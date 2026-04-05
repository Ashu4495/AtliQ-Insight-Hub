import os
import time
import logging
from typing import List, Dict, Any, Tuple

from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain_core.messages import HumanMessage, AIMessage

# --- Path adjustment to allow importing from src ---
import sys
current_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.abspath(os.path.join(current_dir, "../../../"))
if root_dir not in sys.path:
    sys.path.insert(0, root_dir)

from src.ingestion.vectorstore.chroma_store import load_vectorstore
from src.guardrails.pii_guard import sanitize_input, sanitize_output

# --- Configuration ---
load_dotenv(os.path.join(root_dir, ".env"))
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
MODEL_NAME = "llama-3.3-70b-versatile"
TEMPERATURE = 0.2

# --- Global Shared Resources (Singletons) ---
_shared_vectorstore = None
_shared_llm = None
_cached_chains = {}

# --- RBAC and Permissions ---
ROLE_PERMISSIONS = {
    "employee": ["general"],
    "finance_team": ["finance", "general"],
    "hr_team": ["hr", "general"],
    "marketing_team": ["marketing", "general"],
    "engineering_team": ["engineering", "general"],
    "c_level": ["finance", "hr", "marketing", "engineering", "general"],
}

DEPARTMENT_KEYWORDS = {
    "finance": ["finance", "revenue", "budget", "expense", "profit", "loss", "financial", "salary", "payroll"],
    "hr": ["hr", "human resource", "leave", "holiday", "resign", "notice period", "benefits"],
    "marketing": ["marketing", "campaign", "customer", "brand", "advertisement"],
    "engineering": ["engineering", "software", "architecture", "deployment", "bug", "code"],
    "general": ["policy", "office", "handbook", "company", "guideline", "wifi"],
}

# --- Prompts ---
RECONTEXTUALIZE_SYSTEM_PROMPT = (
    "Given a chat history and the latest user question "
    "which might reference context in the chat history, "
    "formulate a standalone question which can be understood "
    "without the chat history. Do NOT answer the question, "
    "just reformulate it if needed and otherwise return it as is."
)

SYSTEM_PROMPT = (
    "You are AtliQ Insight Hub, an internal assistant for FinSolve Technologies.\n\n"
    "STRICT SECURITY RULES:\n"
    "1. Answer using ONLY the provided context. No hallucination.\n"
    "2. NEVER reveal individual salaries, exact CTC, or specific performance ratings of employees.\n"
    "3. If asked for salary, say: 'I am not authorized to disclose individual compensation details.'\n"
    "4. Format in Markdown — bold for emphasis, bullets for lists.\n"
    "5. If the answer is not in the context, say exactly: 'I don't have that information in our current knowledge base.'\n\n"
    "Current User Context:\n"
    "{user_info}\n\n"
    "Retrieved Context:\n"
    "{context}"
)

# --- Guardrail Functions ---
def _detect_departments(question: str) -> List[str]:
    q = question.lower()
    return [dept for dept, keywords in DEPARTMENT_KEYWORDS.items() if any(kw in q for kw in keywords)]

def is_security_risk(question: str) -> Tuple[bool, str]:
    q = question.lower()
    if any(kw in q for kw in ["salary", "ctc", "earns", "pay"]):
        return True, "Salary Details"
    if any(kw in q for kw in ["rating", "performance review"]):
        return True, "Performance Data"
    return False, ""

def is_out_of_scope(question: str, role: str) -> Tuple[bool, str]:
    allowed = set(ROLE_PERMISSIONS.get(role, ["general"]))
    detected = _detect_departments(question)
    if not detected: return False, ""
    blocked = [d for d in detected if d not in allowed]
    if blocked and len(blocked) == len(detected):
        return True, ", ".join(blocked)
    return False, ""

# --- Factory Functions ---
def _get_llm():
    global _shared_llm
    if _shared_llm is None:
        _shared_llm = ChatGroq(
            api_key=GROQ_API_KEY,
            model_name=MODEL_NAME,
            temperature=TEMPERATURE,
        )
    return _shared_llm

def _get_retriever(role: str):
    global _shared_vectorstore
    if _shared_vectorstore is None:
        _shared_vectorstore = load_vectorstore()
        
    allowed_departments = ROLE_PERMISSIONS.get(role, ["general"])
    return _shared_vectorstore.as_retriever(search_kwargs={
        "k": 10,
        "filter": {"department": {"$in": allowed_departments}},
    })

def build_retrieval_chain(role: str, user_context: Dict[str, Any] = None):
    llm = _get_llm()
    retriever = _get_retriever(role)
    
    # Contextualize question prompt
    contextualize_q_prompt = ChatPromptTemplate.from_messages([
        ("system", RECONTEXTUALIZE_SYSTEM_PROMPT),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ])
    
    history_aware_retriever = create_history_aware_retriever(llm, retriever, contextualize_q_prompt)
    
    # Answer prompt
    user_info = f"- Name: {user_context.get('name', 'Associate')}\n- Dept: {user_context.get('department', 'General')}\n- Role: {user_context.get('role', 'Member')}" if user_context else "Anonymous User"
    
    qa_prompt = ChatPromptTemplate.from_messages([
        ("system", SYSTEM_PROMPT.format(user_info=user_info, context="{context}")),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ])
    
    question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)
    
    return create_retrieval_chain(history_aware_retriever, question_answer_chain)

# --- Public Interface ---
def run_chain(question: str, chat_history: List[Any], role: str = "employee", user_context: Dict[str, Any] = None) -> Dict[str, Any]:
    # 1. Sanitize
    sanitized_question = sanitize_input(question)
    
    # 2. Security Check
    risk, topic = is_security_risk(sanitized_question)
    if risk:
        return {"answer": f"🔒 I cannot disclose **{topic}**.", "source_documents": []}
    
    # 3. Scope check
    out_of_scope, detected = is_out_of_scope(sanitized_question, role)
    if out_of_scope:
        return {"answer": f"⚠️ Access restricted. You asked about **{detected}**.", "source_documents": []}
    
    # 4. Invoke Chain
    chain = build_retrieval_chain(role, user_context)
    
    # Convert chat_history to LangChain messages if they aren't already
    formatted_history = []
    for msg in chat_history:
        if isinstance(msg, dict):
            if msg.get("type") == "human": formatted_history.append(HumanMessage(content=msg["content"]))
            elif msg.get("type") == "ai": formatted_history.append(AIMessage(content=msg["content"]))
        else:
            formatted_history.append(msg)
            
    response = chain.invoke({
        "input": sanitized_question,
        "chat_history": formatted_history
    })
    
    # 5. Sanitize Output
    final_answer = sanitize_output(response["answer"])
    
    return {
        "answer": final_answer,
        "source_documents": response.get("context", [])
    }

if __name__ == "__main__":
    # Test
    res = run_chain("What is the attendance policy?", [], role="employee")
    print(f"AI: {res['answer']}")
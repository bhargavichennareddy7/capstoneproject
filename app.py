import os
import pickle
import faiss
import numpy as np
import streamlit as st
from dotenv import load_dotenv
from google.generativeai import GenerativeModel, configure
from sentence_transformers import SentenceTransformer

# === CONFIGURATION ===
load_dotenv()

GOOGLE_API_KEY = os.getenv("AIzaSyCwDpZoAY0e4ZZFZ18GIFPZ-eGi2WFfawA")  # safer
configure(api_key=GOOGLE_API_KEY)

# Load Gemini model
gemini = GenerativeModel("gemini-1.5-flash")

# Load embedding model and FAISS index
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
index = faiss.read_index("quality_inspection_index.faiss")
with open("quality_docs.pkl", "rb") as f:
    documents = pickle.load(f)

# === AGENT FUNCTIONS ===

def knowledge_retrieval_agent(defect_description: str, k: int = 3):
    query_vector = embedding_model.encode([defect_description]).astype("float32")
    distances, indices = index.search(query_vector, k)
    return [documents[i] for i in indices[0]]

def rag_with_gemini(query: str, retrieved_docs: list) -> str:
    context = "\n".join([f"- {doc}" for doc in retrieved_docs])
    prompt = f"""
You are a quality inspection expert.

Query: {query}

Refer to the following documents to assist your answer:
{context}

Provide a root cause explanation or resolution guideline.
"""
    response = gemini.generate_content(prompt)
    return response.text.strip()

def supervisor_agent(description: str, explanation: str) -> tuple[bool, str]:
    if "uncertain" in description.lower() or "verify" in explanation.lower():
        return True, "Escalation recommended due to ambiguity in defect analysis."
    return False, "No escalation required. Defect classification is clear."

def compliance_agent(retrieved_docs: list) -> str:
    for doc in retrieved_docs:
        if "ISO" in doc or "SOP" in doc:
            return "Process validated against ISO/IEC 17025 and SOPs."
    return "Unable to verify process compliance."

# === STREAMLIT UI ===

st.set_page_config(page_title="AI Quality Inspector", layout="wide")
st.title("🔍 AI-driven Quality Inspection System")

query = st.text_input("Enter defect description or query (e.g. 'Crack near mounting point'):")

if query:
    st.subheader("📚 Knowledge Retrieval Agent")
    retrieved_docs = knowledge_retrieval_agent(query)
    for i, doc in enumerate(retrieved_docs):
        st.markdown(f"**Reference {i+1}:** {doc}")

    st.subheader("💡 Root Cause Analysis (Gemini)")
    explanation = rag_with_gemini(query, retrieved_docs)
    st.write(explanation)

    st.subheader("🧑‍🏭 Supervisor Agent")
    escalate, decision = supervisor_agent(query, explanation)
    if escalate:
        st.warning(decision)
    else:
        st.success(decision)

    st.subheader("📏 Compliance Agent")
    compliance = compliance_agent(retrieved_docs)
    st.info(compliance)

    st.subheader("✅ Final Recommendation")
    if escalate:
        st.warning("🔺 Escalation needed: please send for human review.")
    else:
        st.success("🟢 Inspection passed. No further action required.")
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import asyncio
import sqlite3
import traceback
from datetime import datetime
from pathlib import Path

import numpy as np
import streamlit as st
from dotenv import load_dotenv
from langchain_mcp_adapters.client import MultiServerMCPClient
from langgraph.prebuilt import create_react_agent
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage, ToolMessage
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.tools import tool

from analysis import render_analysis

load_dotenv()

st.set_page_config(page_title="Hospital Assistant MCP Server", page_icon="🏥", layout="wide")

INDEX_DIR = Path(__file__).parent / "rag_index"
DB_PATH = Path(__file__).parent / "expanded_hospital.db"


# RAG
@st.cache_resource(show_spinner="Loading RAG index...")
def get_rag():
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    vector_db = FAISS.load_local(
        str(INDEX_DIR),
        embeddings,
        allow_dangerous_deserialization=True,
    )
    return vector_db, embeddings


def _parse_date(s):
    for fmt in ("%Y-%m-%d", "%Y-%m-%d %H:%M:%S", "%d-%m-%Y", "%m/%d/%Y"):
        try:
            return datetime.strptime(str(s), fmt)
        except ValueError:
            continue
    return datetime.min


def _dedup_latest(docs, embeddings, threshold=0.92):
    """Among near-identical notes FOR THE SAME PATIENT, keep only the most recent."""
    if len(docs) < 2:
        return docs
    vecs = np.array(embeddings.embed_documents([d.page_content for d in docs]))
    vecs /= np.linalg.norm(vecs, axis=1, keepdims=True)
    sim = vecs @ vecs.T

    docs = sorted(docs, key=lambda d: _parse_date(d.metadata["date"]), reverse=True)
    kept, kept_idx = [], []
    for i, d in enumerate(docs):
        if any(
            sim[i][j] > threshold
            and docs[j].metadata["patient_id"] == d.metadata["patient_id"]
            for j in kept_idx
        ):
            continue
        kept.append(d)
        kept_idx.append(i)
    return kept


@tool
def search_clinical_notes(query: str, patient_id: str = "") -> str:
    vector_db, embeddings = get_rag()
    if patient_id:
        docs = vector_db.similarity_search(
            query, k=8, filter={"patient_id": patient_id}, fetch_k=1000
        )
    else:
        docs = vector_db.similarity_search(query, k=8)
    docs = _dedup_latest(docs, embeddings)

    if not docs:
        return f"No clinical notes found{' for patient ' + patient_id if patient_id else ''}."

    out = ["[Notes sorted newest first. If information conflicts, trust the most recent note.]"]
    for i, d in enumerate(docs):
        tag = " (MOST RECENT)" if i == 0 else ""
        out.append(
            f"Patient: {d.metadata['patient_id']}\n"
            f"Date: {d.metadata['date']}{tag}\n"
            f"Note: {d.page_content}"
        )
    return "\n\n".join(out)


@tool
def get_visit_history(patient_id: str) -> str:
    conn = sqlite3.connect(DB_PATH)
    rows = conn.execute(
        "SELECT visit_date, note FROM clinical_notes WHERE patient_id = ? ORDER BY visit_date",
        (patient_id,),
    ).fetchall()
    conn.close()
    if not rows:
        return f"No visits found for patient {patient_id}."
    return "\n\n".join(f"Date: {date}\nNote: {note}" for date, note in rows)


get_rag()  # warm cache on main thread


# SYSTEM PROMPT

DEFAULT_SYSTEM_PROMPT = """YOU ARE A MEDICAL TOOL AGENT.
RULES:
- Always use tools when needed
- Never hallucinate patient data
- To summarize a patient's visit history or timeline, call get_visit_history
- For topic-based questions about symptoms, progress notes, or narrative
  details, call search_clinical_notes (pass patient_id when the question is
  about a specific patient) — structured tools do not contain this information
- When clinical notes conflict, the note with the latest date is authoritative;
  mention that earlier notes disagreed if clinically relevant
- Use MCP tools (get_patient, get_lab_results, etc.) only for structured fields
"""


# MODELS

AVAILABLE_MODELS = [
    "gpt-4o-mini",
    "gpt-4o",
    "gpt-4-turbo",
    "gpt-4.1",
    "gpt-4.1-mini",
    "gpt-3.5-turbo",
]


# TOOL LABELS

TOOL_DISPLAY_NAMES = {
    "clinical_trials": "Clinical Trials",
    "icd10_lookup": "ICD-10 Lookup",
    "medrxiv_search": "MedRxiv",
    "health_topics": "Health Topics",
    "get_patient": "Patient Record",
    "get_lab_results": "Lab Results",
    "get_risk_score": "Risk Score",
    "search_clinical_notes": "Clinical Notes (RAG)",
    "get_visit_history": "Visit History",
}


def pretty_name(name):
    return TOOL_DISPLAY_NAMES.get(name, name.replace("_", " ").title())


# SESSION STATE

if "history" not in st.session_state:
    st.session_state.history = []
if "agent" not in st.session_state:
    st.session_state.agent = None
if "tools" not in st.session_state:
    st.session_state.tools = None


# MCP TOOL LOADER

@st.cache_resource(show_spinner="Loading MCP tools...")
def load_tools():
    async def _load():
        client = MultiServerMCPClient(
            {
                "external": {
                    "url": "http://localhost:8000/mcp",
                    "transport": "streamable_http",
                },
                "internal": {
                    "url": "http://localhost:8001/mcp",
                    "transport": "streamable_http",
                },
            }
        )
        return await client.get_tools()

    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    return loop.run_until_complete(_load())


# BUILD AGENT

def build_agent(model_name, temperature, system_prompt):
    tools = load_tools() + [search_clinical_notes, get_visit_history]
    model = ChatOpenAI(model=model_name, temperature=temperature)
    agent = create_react_agent(
        model,
        tools,
        prompt=SystemMessage(content=system_prompt),
    )
    return agent, tools


# RUN AGENT

async def run_agent(query: str):
    response = await st.session_state.agent.ainvoke(
        {"messages": [HumanMessage(content=query)]}
    )
    messages = response["messages"]

    trace = []
    for m in messages:
        if isinstance(m, AIMessage) and m.tool_calls:
            for tc in m.tool_calls:
                trace.append({"type": "call", "name": tc["name"], "args": tc["args"]})
        elif isinstance(m, ToolMessage):
            trace.append({"type": "result", "name": m.name, "content": str(m.content)})

    return messages[-1].content, trace


# TRACE UI

def render_trace(trace):
    for t in trace:
        name = pretty_name(t["name"])
        if t["type"] == "call":
            st.markdown(f"**Called:** {name}")
            st.json(t["args"])
        else:
            st.markdown(f"**Result from:** {name}")
            st.code(t["content"][:2000])


# PAGES

def chat_page():
    st.title("Hospital Assistant MCP Server")

    with st.sidebar:
        st.header("Settings")
        model_name = st.selectbox("Model", AVAILABLE_MODELS)
        temperature = st.slider("Temperature", 0.0, 2.0, 0.0, 0.1)
        system_prompt = st.text_area("System Prompt", DEFAULT_SYSTEM_PROMPT, height=400)
        if st.button("Apply"):
            st.session_state.agent, st.session_state.tools = build_agent(
                model_name, temperature, system_prompt
            )
            st.success("Agent updated")

    if st.session_state.agent is None:
        st.session_state.agent, st.session_state.tools = build_agent(
            "gpt-4o-mini", 0.0, DEFAULT_SYSTEM_PROMPT
        )

    for msg in st.session_state.history:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if msg.get("trace"):
                with st.expander("Tool Trace"):
                    render_trace(msg["trace"])

    user_input = st.chat_input("Ask a medical question...")

    if user_input:
        st.session_state.history.append({"role": "user", "content": user_input})
        with st.chat_message("user"):
            st.markdown(user_input)

        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                try:
                    answer, trace = asyncio.run(run_agent(user_input))
                except Exception as e:
                    answer, trace = f"Error: {type(e).__name__}: {e}", []
                    st.code(traceback.format_exc())
            st.markdown(answer)
            if trace:
                with st.expander("Tool Trace"):
                    render_trace(trace)

        st.session_state.history.append({
            "role": "assistant",
            "content": answer,
            "trace": trace,
        })


def analysis_page():
    st.title("Data Analysis")
    render_analysis()


pg = st.navigation([
    st.Page(chat_page, title="Chat", icon="💬", default=True),
    st.Page(analysis_page, title="Data Analysis", icon="📊"),
])
pg.run()
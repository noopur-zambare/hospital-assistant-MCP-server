# 🏥 Hospital Assistant MCP Server

A dual MCP-server system built with a **LangGraph ReAct agent** and a **Streamlit UI**, combining external medical knowledge APIs with a local patient database for clinical lookups, ICD-10 coding, and research-backed responses.

---

## 🧩 System Overview

The project runs **three components**:

1. **External MCP Server** (port `8000`) - public medical knowledge APIs
2. **Internal MCP Server** (port `8001`) - local patient database
3. **Streamlit App** - chat UI with an OpenAI agent that calls tools across both servers

---

## 🌐 External MCP Server

Exposes public medical APIs as MCP tools.

### Integrated APIs

| API | Purpose |
|-----|---------|
| Health.gov MyHealthfinder | Consumer health topics |
| NLM Clinical Tables (ICD-10-CM) | Diagnosis code lookup |
| ClinicalTrials.gov v2 | Active and completed trials |
| medRxiv | Recent medical research preprints |

### Tools

#### `health_topics`
Search Health.gov MyHealthfinder for consumer health information by keyword.

#### `icd10_lookup`
Look up ICD-10-CM codes by name or keyword via the NLM Clinical Tables API.

#### `clinical_trials`
Search ClinicalTrials.gov v2 for studies by condition.

#### `medrxiv_search`
Pull recent medRxiv preprints matching a query (180-day window).

---

## 🏥 Internal MCP Server

Serves a local patient database. All patient data stays on-prem.

### Tools

#### `get_patient`
Fetch a patient's record by ID (e.g. `P003`).

#### `get_lab_results`
Retrieve a patient's lab test results.

#### `get_risk_score`
Get a patient's calculated clinical risk score.

#### `list_patients`
List all patients in the internal database.

---

## 💬 Streamlit Agent UI

A chat interface that wires both MCP servers into a LangGraph ReAct agent.

### Features

- **Multi-server MCP client** - talks to both external (8000) and internal (8001) servers in one agent.
- **Tool trace per turn** - every assistant reply has an expandable panel showing exactly which tools were called, with what arguments, and what they returned.
- **Structured tool outputs** - every response is grounded in tool results, fully visible in the trace.
- **Decision support, not diagnosis** - designed to help clinicians look up information, not to replace clinical judgement.
- **Settings panel** - switch OpenAI model, adjust temperature, and edit the system prompt live without restarting.

---

## ⚙️ Architecture

```
hospital-assistant-MCP-server/
│
├── server/
│   ├── external_server.py  # MCP server on :8000
│   │                       #   - health_topics
│   │                       #   - icd10_lookup
│   │                       #   - clinical_trials
│   │                       #   - medrxiv_search
│   │
│   └── internal_server.py  # MCP server on :8001
│                           #   - get_patient
│                           #   - get_lab_results
│                           #   - get_risk_score
│                           #   - list_patients
│
├── app.py                  # Streamlit chat UI + LangGraph agent
├── .env                    # OPENAI_API_KEY, etc.
├── hospital.db             # synthetic SQL database
└── README.md
```

---

## 🚀 Running

### 1. Install dependencies

```bash
python -m venv mcp-env
source mcp-env/bin/activate
pip install streamlit langchain-mcp-adapters langgraph langchain-openai \
            python-dotenv httpx mcp
```

### 2. Set environment variables

Create a `.env` file:

```
OPENAI_API_KEY=sk-...
```

### 3. Start the MCP servers

In two separate terminals:

```bash
python server/external_server.py    # listens on :8000
```

```bash
python server/internal_server.py    # listens on :8001
```

### 4. Launch the UI

```bash
streamlit run app.py
```

Open the URL Streamlit prints (typically `http://localhost:8501`).

Each response includes a "Tool trace" expander so you can see exactly which tools fired and what they returned.

# Hospital Assistant MCP Server

A dual MCP-server system built with a LangGraph ReAct agent, combining external medical knowledge APIs with a local patient database for clinical lookups, ICD-10 coding, and research-backed responses.

---
### Demo
https://github.com/user-attachments/assets/7ac75e7d-74b0-4f79-aab3-7788b9911056


---
### System Overview

The project runs four components:

1. **External MCP Server** - public medical knowledge APIs
2. **Internal MCP Server** - local patient database
3. **RAG Layer (FAISS)** - unstructured clinical notes retrieval
4. **Streamlit App** - chat UI with an OpenAI agent that calls tools across both servers

---

### External MCP Server

Exposes public medical APIs as MCP tools.

#### Integrated APIs

| API | Purpose |
|-----|---------|
| Health.gov MyHealthfinder | Consumer health topics |
| NLM Clinical Tables (ICD-10-CM) | Diagnosis code lookup |
| ClinicalTrials.gov v2 | Active and completed trials |
| medRxiv | Recent medical research preprints |

#### Tools

- #### health_topics
Search Health.gov MyHealthfinder for consumer health information by keyword.

- #### icd10_lookup
Look up ICD-10-CM codes by name or keyword via the NLM Clinical Tables API.

- #### clinical_trials
Search ClinicalTrials.gov v2 for studies by condition.

- #### medrxiv_search
Pull recent medRxiv preprints matching a query (180-day window).

---

### Internal MCP Server

Serves a local patient database.

### Tools

- #### get_patient
Fetch a patient's record by ID (e.g. `P003`).

- #### get_lab_results
Retrieve a patient's lab test results.

- #### get_risk_score
Get a patient's calculated clinical risk score.

- #### list_patients
List all patients in the internal database.

---
### Features

- **Multi-server MCP client** - talks to both external (8000) and internal (8001) servers in one agent.
- **Tool trace per turn** - every assistant reply has an expandable panel showing exactly which tools were called, with what arguments, and what they returned.
- **Structured tool outputs** - every response is grounded in tool results, fully visible in the trace.
- **Decision support, not diagnosis** - designed to help clinicians look up information, not to replace clinical judgement.
- **Settings panel** - switch OpenAI model, adjust temperature, and edit the system prompt live without restarting.

---

### Architecture

```
hospital-assistant-MCP-server/
│
├── servers/
│   ├── external_server.py   
│   ├── internal_server.py     
│   └── rag_tool.py  
├── app.py                         
├── .env                   
├── hospital.db
└── README.md
```

---

### Running

#### 1. Install dependencies

```bash
python -m venv mcp-env
source mcp-env/bin/activate
pip install streamlit langchain-mcp-adapters langgraph langchain-openai \
            python-dotenv httpx mcp
```

#### 2. Set environment variables

Create a `.env` file:

```
OPENAI_API_KEY=''
```

#### 3. Start the MCP servers

In two separate terminals:

```bash
python servers/external_server.py    
```

```bash
python servers/internal_server.py   
```

#### 4. Launch the UI

```bash
streamlit run app.py
```

---

### Tech Stack
- MCP Servers
- LangGraph
- LangChain
- Streamlit

---

### ⚠️ Disclaimer
This project is for research and educational purposes only.
It is not a substitute for professional medical advice, diagnosis, or treatment.


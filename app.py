import asyncio
import streamlit as st
from langchain_mcp_adapters.client import MultiServerMCPClient
from langgraph.prebuilt import create_react_agent
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage, ToolMessage
from dotenv import load_dotenv

load_dotenv()

st.set_page_config(page_title="Hospital Assistant MCP Server", page_icon="🏥", layout="wide")
st.title("Hospital Assistant MCP Server")

DEFAULT_SYSTEM_PROMPT = """YOU ARE A MEDICAL TOOL-DRIVEN AGENT.

RULES:
- You MUST use tools when available.
- NEVER guess medical or patient data.
- ALWAYS show which tool you used in reasoning.
- For patient IDs, internal database tools MUST be used first.
- For ICD-10, use ICD-10 tool.
- For research, use ClinicalTrials or MedRxiv tools.

OUTPUT FORMAT:
- Tool used (exact name)
- Observations from tool
- Final answer"""

AVAILABLE_MODELS = [
    "gpt-4o-mini",
    "gpt-4o",
    "gpt-4-turbo",
    "gpt-4.1",
    "gpt-4.1-mini",
    "gpt-3.5-turbo",
]

TOOL_DISPLAY_NAMES = {
    "clinical_trials": "Clinical Trials",
    "icd10_lookup": "ICD-10 Lookup",
    "medrxiv_search": "MedRxiv Search",
    "health_topics": "Health Topics",
    "get_patient": "Patient Record",
    "get_lab_results": "Lab Results",
    "get_risk_score": "Risk Score",
    "list_patients": "Patient List",
}


def pretty_name(raw: str) -> str:
    if raw in TOOL_DISPLAY_NAMES:
        return TOOL_DISPLAY_NAMES[raw]
    return raw.replace("_", " ").title()


if "system_prompt" not in st.session_state:
    st.session_state.system_prompt = DEFAULT_SYSTEM_PROMPT

if "model_name" not in st.session_state:
    st.session_state.model_name = "gpt-4o-mini"

if "temperature" not in st.session_state:
    st.session_state.temperature = 0.0

if "history" not in st.session_state:
    st.session_state.history = []

if "pending_query" not in st.session_state:
    st.session_state.pending_query = None


@st.cache_resource(show_spinner="Connecting to MCP servers...")
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

    return asyncio.run(_load())


@st.cache_resource(show_spinner="Building agent...")
def build_agent(model_name: str, temperature: float, system_prompt: str):
    tools = load_tools()
    model = ChatOpenAI(model=model_name, temperature=temperature)
    agent = create_react_agent(
        model,
        tools,
        prompt=SystemMessage(content=system_prompt),
    )
    return agent, tools


with st.sidebar:
    st.header("⚙️ Settings")

    with st.expander("🤖 Model", expanded=True):
        model_name = st.selectbox(
            "Model",
            AVAILABLE_MODELS,
            index=AVAILABLE_MODELS.index(st.session_state.model_name)
            if st.session_state.model_name in AVAILABLE_MODELS
            else 0,
        )
        temperature = st.slider("Temperature", 0.0, 2.0, st.session_state.temperature, 0.1)

    with st.expander("📝 System Prompt", expanded=False):
        system_prompt = st.text_area(
            "Prompt",
            value=st.session_state.system_prompt,
            height=300,
            label_visibility="collapsed",
        )
        col_a, col_b = st.columns(2)
        if col_a.button("💾 Apply", use_container_width=True):
            st.session_state.system_prompt = system_prompt
            st.session_state.model_name = model_name
            st.session_state.temperature = temperature
            st.success("Settings applied")
            st.rerun()
        if col_b.button("↺ Reset", use_container_width=True):
            st.session_state.system_prompt = DEFAULT_SYSTEM_PROMPT
            st.session_state.model_name = "gpt-4o-mini"
            st.session_state.temperature = 0.0
            st.rerun()

    settings_changed = (
        model_name != st.session_state.model_name
        or temperature != st.session_state.temperature
        or system_prompt != st.session_state.system_prompt
    )

    if settings_changed:
        st.warning("Unsaved changes — click Apply.")

    st.session_state.model_name = model_name
    st.session_state.temperature = temperature
    st.session_state.system_prompt = system_prompt

    agent, tools = build_agent(
        st.session_state.model_name,
        st.session_state.temperature,
        st.session_state.system_prompt,
    )

    st.divider()

    st.header("🔧 Loaded Tools")
    st.caption(f"{len(tools)} tools available")
    for t in tools:
        with st.expander(pretty_name(t.name)):
            st.caption(f"`{t.name}`")
            st.write(t.description or "_No description_")

    st.divider()

    st.subheader("💡 Try asking")
    examples = [
        "Show lab results and risk score for patient P003",
        "Patient P002 has high glucose. What condition does this suggest?",
        "Are there clinical trials for pancreatic cancer in Canada?",
        "What is ICD-10 code for pneumonia?",
    ]
    for ex in examples:
        if st.button(ex, key=f"ex_{ex}", use_container_width=True):
            st.session_state.pending_query = ex
            st.rerun()

    st.divider()
    if st.button("🗑 Clear chat", use_container_width=True):
        st.session_state.history = []
        st.rerun()


async def run_agent(query: str):
    response = await agent.ainvoke(
        {"messages": [HumanMessage(content=query)]}
    )

    messages = response["messages"]
    trace = []
    for m in messages:
        if isinstance(m, AIMessage) and m.tool_calls:
            for tc in m.tool_calls:
                trace.append({
                    "type": "call",
                    "name": tc["name"],
                    "args": tc["args"],
                })
        elif isinstance(m, ToolMessage):
            trace.append({
                "type": "result",
                "name": m.name,
                "content": str(m.content),
            })

    final_answer = messages[-1].content
    return final_answer, trace


def render_trace(trace):
    if not trace:
        st.info("No tools were called for this response.")
        return
    for step in trace:
        display = pretty_name(step["name"])
        if step["type"] == "call":
            st.markdown(f"**Called** {display}")
            st.json(step["args"])
        else:
            st.markdown(f"**Result from** {display}")
            content = step["content"]
            if len(content) > 2000:
                content = content[:2000] + "\n\n... (truncated)"
            st.code(content, language="json")


st.caption(
    f"**Model:** `{st.session_state.model_name}` · "
    f"**Temp:** `{st.session_state.temperature}` · "
    f"**Tools:** {len(tools)}"
)

for turn in st.session_state.history:
    with st.chat_message(turn["role"]):
        st.markdown(turn["content"])
        if turn["role"] == "assistant" and turn.get("trace"):
            with st.expander(f"🔍 Tool trace ({len(turn['trace'])} steps)"):
                render_trace(turn["trace"])


user_input = st.chat_input("Ask a medical question...")

if st.session_state.pending_query:
    user_input = st.session_state.pending_query
    st.session_state.pending_query = None

if user_input:
    st.session_state.history.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            try:
                answer, trace = asyncio.run(run_agent(user_input))
            except Exception as e:
                answer = f"Error: {e}"
                trace = []

        st.markdown(answer)
        if trace:
            with st.expander(f"🔍 Tool trace ({len(trace)} steps)"):
                render_trace(trace)

    st.session_state.history.append({
        "role": "assistant",
        "content": answer,
        "trace": trace,
    })
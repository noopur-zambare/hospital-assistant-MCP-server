from langchain_mcp_adapters.client import MultiServerMCPClient
from langgraph.prebuilt import create_react_agent
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage
from dotenv import load_dotenv
import asyncio

load_dotenv()


async def main():

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

    tools = await client.get_tools()

    print("\nTOOLS LOADED:", len(tools))
    print("\nAVAILABLE TOOLS:")
    for t in tools:
        print("-", t.name)

    model = ChatOpenAI(
        model="gpt-4o-mini",
        temperature=0
    ).bind_tools(tools)

    system_prompt = SystemMessage(content="""
YOU ARE A MEDICAL TOOL-DRIVEN AGENT.

RULES:
- You MUST use tools when available.
- NEVER guess medical or patient data.
- ALWAYS show which tool you used in reasoning.
- For patient IDs → internal database tools MUST be used first.
- For ICD-10 → use ICD-10 tool.
- For research → use ClinicalTrials or MedRxiv tools.

OUTPUT FORMAT:
- Tool used (exact name)
- Observations from tool
- Final answer
""")

    agent = create_react_agent(
        model,
        tools,
        prompt=system_prompt
    )

    async def run_query(query: str):

        response = await agent.ainvoke(
            {
                "messages": [
                    system_prompt,
                    HumanMessage(content=query)
                ]
            }
        )

        print("\n" + "=" * 60)
        print("QUERY:", query)

        print("\nFINAL ANSWER:\n")
        print(response["messages"][-1].content)

        print("\nTOOL TRACE (REAL MESSAGES):\n")

        messages = response["messages"]

        tool_used = False

        for m in messages:
            msg_type = getattr(m, "type", "")
            name = getattr(m, "name", None)

            if msg_type == "tool":
                tool_used = True
                print(f"[TOOL CALL] {name}")
                print(m.content)
                print("-" * 40)

        if not tool_used:
            print("No tool messages detected in trace")

        print("=" * 60 + "\n")

        return response["messages"][-1].content

    await run_query("Show lab results and risk score for patient P003")
    # await run_query("Patient P002 has high glucose. What condition does this suggest?")
    # await run_query("Are there clinical trials for pancreatic cancer in Canada?")
    # await run_query("What is ICD-10 code for pneumonia?")

if __name__ == "__main__":
    asyncio.run(main())
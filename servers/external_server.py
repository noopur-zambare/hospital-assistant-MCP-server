from mcp.server.fastmcp import FastMCP
import httpx

mcp = FastMCP("HospitalMCPServer", host="0.0.0.0", port=8000)

HEALTH_BASE = "https://odphp.health.gov/myhealthfinder/api/v4"

@mcp.tool()
async def health_topics(topic: str, language: str = "en"):
    if not topic:
        return {"error": "topic required"}

    async with httpx.AsyncClient() as client:
        url = f"{HEALTH_BASE}/topicsearch.json"
        r = await client.get(url, params={"keyword": topic, "lang": language})
        return r.json()


ICD_BASE = "https://clinicaltables.nlm.nih.gov/api/icd10cm/v3/search"

@mcp.tool()
async def icd10_lookup(query: str, max_results: int = 10):
    async with httpx.AsyncClient() as client:
        r = await client.get(
            ICD_BASE,
            params={"sf": "code,name", "terms": query, "maxList": max_results}
        )
        data = r.json()

    return {"query": query, "results": data}


TRIALS_BASE = "https://clinicaltrials.gov/api/v2/studies"

@mcp.tool()
async def clinical_trials(condition: str, max_results: int = 10):
    async with httpx.AsyncClient() as client:
        r = await client.get(
            TRIALS_BASE,
            params={"query.cond": condition, "pageSize": max_results, "format": "json"}
        )
        return r.json()


MEDRXIV_BASE = "https://api.medrxiv.org/"

@mcp.tool()
async def medrxiv_search(query: str, max_results: int = 10):
    url = f"{MEDRXIV_BASE}details/medrxiv/{query}/0/180/json"

    async with httpx.AsyncClient() as client:
        r = await client.get(url)
        data = r.json()

    return {"query": query, "results": data.get("collection", [])[:max_results]}


if __name__ == "__main__":
    mcp.run(transport="streamable-http")
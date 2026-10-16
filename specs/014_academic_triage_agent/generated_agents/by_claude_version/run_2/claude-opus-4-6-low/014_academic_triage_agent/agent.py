import os
import urllib.parse
import urllib.request
import json
from typing import TypedDict, Annotated, Literal

import xmltodict
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage, ToolMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, END


# ── Tools ──

@tool
def search_arxiv(query: str) -> str:
    """Search for academic papers on arXiv. Used for Exact Sciences/Technology domain.
    Args:
        query: The search query for arXiv papers.
    Returns:
        Raw data about papers found on arXiv.
    """
    url = f"http://export.arxiv.org/api/query?search_query=all:{urllib.parse.quote(query)}&start=0&max_results=5"
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "AcademicTriageAgent/1.0"})
        with urllib.request.urlopen(req, timeout=30) as resp:
            data = xmltodict.parse(resp.read().decode())
        entries = data.get("feed", {}).get("entry", [])
        if not entries:
            return "No results found."
        if isinstance(entries, dict):
            entries = [entries]
        results = []
        for e in entries:
            title = e.get("title", "").replace("\n", " ").strip()
            summary = e.get("summary", "").replace("\n", " ").strip()
            results.append(f"Title: {title}\nSummary: {summary}")
        return "\n---\n".join(results)
    except Exception as exc:
        return f"Error searching arXiv: {exc}"


@tool
def search_pubmed(query: str) -> str:
    """Search for academic papers on PubMed. Used for Life Sciences/Health domain.
    Args:
        query: The search query for PubMed papers.
    Returns:
        Raw data about papers found on PubMed.
    """
    try:
        search_url = (
            "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi?"
            f"db=pubmed&term={urllib.parse.quote(query)}&retmax=5&retmode=json"
        )
        req = urllib.request.Request(search_url, headers={"User-Agent": "AcademicTriageAgent/1.0"})
        with urllib.request.urlopen(req, timeout=30) as resp:
            search_data = json.loads(resp.read().decode())
        ids = search_data.get("esearchresult", {}).get("idlist", [])
        if not ids:
            return "No results found."
        fetch_url = (
            "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esummary.fcgi?"
            f"db=pubmed&id={','.join(ids)}&retmode=json"
        )
        req2 = urllib.request.Request(fetch_url, headers={"User-Agent": "AcademicTriageAgent/1.0"})
        with urllib.request.urlopen(req2, timeout=30) as resp2:
            fetch_data = json.loads(resp2.read().decode())
        results = []
        for pid in ids:
            paper = fetch_data.get("result", {}).get(pid, {})
            title = paper.get("title", "Unknown")
            source = paper.get("source", "")
            pubdate = paper.get("pubdate", "")
            results.append(f"Title: {title}\nSource: {source}\nDate: {pubdate}")
        return "\n---\n".join(results)
    except Exception as exc:
        return f"Error searching PubMed: {exc}"


@tool
def get_wikipedia_summary(topic: str) -> str:
    """Retrieve a summary of a topic from Wikipedia. Used for General Concepts domain.
    Args:
        topic: The topic to look up on Wikipedia.
    Returns:
        Raw summary data from Wikipedia.
    """
    try:
        url = (
            "https://en.wikipedia.org/api/rest_v1/page/summary/"
            + urllib.parse.quote(topic.replace(" ", "_"))
        )
        req = urllib.request.Request(url, headers={"User-Agent": "AcademicTriageAgent/1.0"})
        with urllib.request.urlopen(req, timeout=30) as resp:
            data = json.loads(resp.read().decode())
        extract = data.get("extract", "")
        if not extract:
            return "No summary found."
        return f"Topic: {data.get('title', topic)}\nSummary: {extract}"
    except Exception as exc:
        return f"Error fetching Wikipedia summary: {exc}"


# ── State ──

class AgentState(TypedDict):
    user_request: str
    result: str


# ── Graph nodes ──

TOOLS = [search_arxiv, search_pubmed, get_wikipedia_summary]
TOOL_MAP = {t.name: t for t in TOOLS}

SYSTEM_PROMPT = """You are an academic triage agent. Your job is to analyze the user's request and decide the correct action.

Rules:
1. If the input is empty or whitespace-only, respond EXACTLY: Error: The user request cannot be empty.
2. If the request spans multiple distinct domains (e.g., asks about both AI and medicine in one query), respond EXACTLY: Error: The query spans multiple distinct domains. Please make one specific research request at a time.
3. If the request is not academic (personal advice, transactions, creative writing, etc.), respond EXACTLY: Error: Query outside of academic scope.
4. If the request explicitly asks for papers/research in Exact Sciences/Technology (e.g., computer science, mathematics, physics, engineering, artificial intelligence), call search_arxiv with an appropriate query.
5. If the request explicitly asks for papers/research in Life Sciences/Health (e.g., biology, medicine, genetics, melanoma), call search_pubmed with an appropriate query.
6. If the request asks to explain or define a concept (even a scientific one), WITHOUT explicitly asking for papers or research, call get_wikipedia_summary with the concept.
7. Call exactly ONE tool per request. Never call multiple tools.

You must decide which single action to take based on the user's request."""


def classify_and_act(state: AgentState) -> AgentState:
    user_request = state["user_request"]

    if not user_request or not user_request.strip():
        return {"user_request": user_request, "result": "Error: The user request cannot be empty."}

    try:
        llm = ChatOpenAI(model="gpt-4o", temperature=0)
        llm_with_tools = llm.bind_tools(TOOLS)

        messages = [
            SystemMessage(content=SYSTEM_PROMPT),
            HumanMessage(content=user_request),
        ]

        response = llm_with_tools.invoke(messages)

        if isinstance(response.content, str) and response.content.startswith("Error:"):
            return {"user_request": user_request, "result": response.content.strip()}

        if not response.tool_calls:
            content = response.content.strip() if response.content else ""
            for err_msg in [
                "Error: The user request cannot be empty.",
                "Error: The query spans multiple distinct domains. Please make one specific research request at a time.",
                "Error: Query outside of academic scope.",
            ]:
                if err_msg in content:
                    return {"user_request": user_request, "result": err_msg}
            if content:
                return {"user_request": user_request, "result": content}
            return {"user_request": user_request, "result": "Error: Unable to get response from agent."}

        tool_call = response.tool_calls[0]
        tool_name = tool_call["name"]
        tool_args = tool_call["args"]

        if tool_name not in TOOL_MAP:
            return {"user_request": user_request, "result": "Error: Unable to get response from agent."}

        tool_result = TOOL_MAP[tool_name].invoke(tool_args)

        if not tool_result or "No results found" in str(tool_result) or "Error" in str(tool_result):
            return {"user_request": user_request, "result": "Error: Unable to get response from agent."}

        messages.append(response)
        messages.append(ToolMessage(content=str(tool_result), tool_call_id=tool_call["id"]))

        synthesis_msg = SystemMessage(
            content="Based on the tool results, synthesize a response. "
            "If the tool was search_arxiv or search_pubmed, provide a brief introduction followed by a list of papers with titles and short summaries. "
            "If the tool was get_wikipedia_summary, provide a coherent explanatory paragraph about the concept. "
            "Base your response EXCLUSIVELY on the data returned by the tool. Do not invent any information."
        )
        messages.append(synthesis_msg)

        final_response = llm.invoke(messages)
        final_content = final_response.content.strip() if final_response.content else ""

        if not final_content:
            return {"user_request": user_request, "result": "Error: Unable to get response from agent."}

        return {"user_request": user_request, "result": final_content}

    except Exception:
        return {"user_request": user_request, "result": "Error: Unable to get response from agent."}


# ── Graph ──

def build_graph():
    graph = StateGraph(AgentState)
    graph.add_node("classify_and_act", classify_and_act)
    graph.set_entry_point("classify_and_act")
    graph.add_edge("classify_and_act", END)
    return graph.compile()


GRAPH = build_graph()


def agent(user_request: str) -> str:
    result = GRAPH.invoke({"user_request": user_request, "result": ""})
    return result["result"]


if __name__ == "__main__":
    import sys
    query = " ".join(sys.argv[1:]) if len(sys.argv) > 1 else "Find papers about machine learning"
    print(agent(query))

import os
import urllib.request
import urllib.parse
import json
from typing import TypedDict, Annotated
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage, ToolMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, END
import xmltodict


@tool
def search_arxiv(query: str) -> str:
    """Searches for papers on arXiv and returns raw data about the papers found. Used for Exact Sciences/Technology domain."""
    try:
        encoded = urllib.parse.quote(query)
        url = f"http://export.arxiv.org/api/query?search_query=all:{encoded}&start=0&max_results=5"
        req = urllib.request.Request(url, headers={"User-Agent": "AcademicTriageAgent/1.0"})
        with urllib.request.urlopen(req, timeout=15) as resp:
            data = resp.read().decode("utf-8")
        parsed = xmltodict.parse(data)
        feed = parsed.get("feed", {})
        entries = feed.get("entry", [])
        if not entries:
            return "No results found."
        if isinstance(entries, dict):
            entries = [entries]
        results = []
        for e in entries:
            title = e.get("title", "Unknown").replace("\n", " ").strip()
            summary = e.get("summary", "No summary.").replace("\n", " ").strip()
            results.append(f"Title: {title}\nSummary: {summary}")
        return "\n\n".join(results)
    except Exception as e:
        return f"Error searching arXiv: {e}"


@tool
def search_pubmed(query: str) -> str:
    """Searches for papers on PubMed and returns raw data about the papers found. Used for Life Sciences/Health domain."""
    try:
        encoded = urllib.parse.quote(query)
        search_url = f"https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi?db=pubmed&term={encoded}&retmax=5&retmode=json"
        req = urllib.request.Request(search_url, headers={"User-Agent": "AcademicTriageAgent/1.0"})
        with urllib.request.urlopen(req, timeout=15) as resp:
            search_data = json.loads(resp.read().decode("utf-8"))
        id_list = search_data.get("esearchresult", {}).get("idlist", [])
        if not id_list:
            return "No results found."
        ids = ",".join(id_list)
        fetch_url = f"https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esummary.fcgi?db=pubmed&id={ids}&retmode=json"
        req2 = urllib.request.Request(fetch_url, headers={"User-Agent": "AcademicTriageAgent/1.0"})
        with urllib.request.urlopen(req2, timeout=15) as resp2:
            fetch_data = json.loads(resp2.read().decode("utf-8"))
        result_entries = fetch_data.get("result", {})
        results = []
        for pid in id_list:
            entry = result_entries.get(pid, {})
            title = entry.get("title", "Unknown")
            source = entry.get("source", "")
            pubdate = entry.get("pubdate", "")
            results.append(f"Title: {title}\nSource: {source}\nDate: {pubdate}")
        return "\n\n".join(results)
    except Exception as e:
        return f"Error searching PubMed: {e}"


@tool
def get_wikipedia_summary(topic: str) -> str:
    """Retrieves a summary of a topic from Wikipedia and returns the raw summary data. Used for General Concepts domain."""
    try:
        encoded = urllib.parse.quote(topic)
        url = f"https://en.wikipedia.org/api/rest_v1/page/summary/{encoded}"
        req = urllib.request.Request(url, headers={"User-Agent": "AcademicTriageAgent/1.0"})
        with urllib.request.urlopen(req, timeout=15) as resp:
            data = json.loads(resp.read().decode("utf-8"))
        title = data.get("title", topic)
        extract = data.get("extract", "No summary available.")
        return f"Title: {title}\nSummary: {extract}"
    except Exception as e:
        return f"Error fetching Wikipedia summary: {e}"


TOOLS = [search_arxiv, search_pubmed, get_wikipedia_summary]
TOOL_MAP = {t.name: t for t in TOOLS}

SYSTEM_PROMPT = """You are an academic triage agent. Your job is to analyze a user's academic request and select exactly one appropriate tool to answer it.

RULES:
1. If the input is empty or whitespace-only, respond with exactly: Error: The user request cannot be empty.
2. If the request is NOT academic (personal advice, transactions, creative writing, etc.), respond with exactly: Error: Query outside of academic scope.
3. If the request spans multiple distinct domains (e.g., papers about AI AND melanoma), respond with exactly: Error: The query spans multiple distinct domains. Please make one specific research request at a time.
4. For a valid academic request:
   - If explicitly asking for papers/research in Exact Sciences/Technology (e.g., AI, physics, math, CS, engineering): call search_arxiv
   - If explicitly asking for papers/research in Life Sciences/Health (e.g., medicine, biology, genetics): call search_pubmed
   - If asking to explain/define a concept (even if from a scientific domain, as long as not explicitly asking for papers): call get_wikipedia_summary
5. Call EXACTLY ONE tool, never more.
6. Base your final answer ONLY on tool results. Never invent papers or information.

For papers: provide a brief introduction followed by a list of papers with titles and short summaries.
For concepts: provide a coherent explanatory paragraph.
"""


class AgentState(TypedDict):
    messages: list
    result: str


def classify_node(state: AgentState) -> AgentState:
    llm = ChatOpenAI(model="gpt-4o", temperature=0)
    llm_with_tools = llm.bind_tools(TOOLS)
    messages = state["messages"]
    response = llm_with_tools.invoke(messages)
    return {"messages": messages + [response], "result": ""}


def should_continue(state: AgentState) -> str:
    last = state["messages"][-1]
    if hasattr(last, "tool_calls") and last.tool_calls:
        return "tool"
    return "respond"


def tool_node(state: AgentState) -> AgentState:
    messages = state["messages"]
    last = messages[-1]
    new_messages = list(messages)
    for tc in last.tool_calls:
        tool_name = tc["name"]
        tool_args = tc["args"]
        t = TOOL_MAP.get(tool_name)
        if t:
            try:
                result = t.invoke(tool_args)
            except Exception as e:
                result = f"Error: {e}"
        else:
            result = "Error: Unknown tool."
        new_messages.append(ToolMessage(content=result, tool_call_id=tc["id"]))
    return {"messages": new_messages, "result": ""}


def synthesize_node(state: AgentState) -> AgentState:
    llm = ChatOpenAI(model="gpt-4o", temperature=0)
    messages = state["messages"]
    response = llm.invoke(messages)
    return {"messages": messages + [response], "result": response.content}


def respond_node(state: AgentState) -> AgentState:
    last = state["messages"][-1]
    return {"messages": state["messages"], "result": last.content}


def build_graph():
    g = StateGraph(AgentState)
    g.add_node("classify", classify_node)
    g.add_node("tool", tool_node)
    g.add_node("synthesize", synthesize_node)
    g.add_node("respond", respond_node)
    g.set_entry_point("classify")
    g.add_conditional_edges("classify", should_continue, {"tool": "tool", "respond": "respond"})
    g.add_edge("tool", "synthesize")
    g.add_edge("synthesize", END)
    g.add_edge("respond", END)
    return g.compile()


def agent(user_request: str) -> str:
    if not user_request or not user_request.strip():
        return "Error: The user request cannot be empty."
    try:
        graph = build_graph()
        messages = [
            SystemMessage(content=SYSTEM_PROMPT),
            HumanMessage(content=user_request),
        ]
        result = graph.invoke({"messages": messages, "result": ""})
        output = result.get("result", "")
        if not output or not output.strip():
            return "Error: Unable to get response from agent."
        return output
    except Exception:
        return "Error: Unable to get response from agent."


if __name__ == "__main__":
    import sys
    query = " ".join(sys.argv[1:]) if len(sys.argv) > 1 else "Find papers about machine learning"
    print(agent(query))

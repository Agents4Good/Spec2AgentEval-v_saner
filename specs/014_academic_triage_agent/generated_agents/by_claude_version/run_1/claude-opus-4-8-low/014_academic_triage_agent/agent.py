"""Academic Triage Agent.

Triages a user's academic request, selects exactly one appropriate tool based
on the subject domain (Exact Sciences/Technology, Life Sciences/Health, or
General Concepts) and returns a formatted summary of the results found, using
an LLM (GPT-4o via ChatOpenAI) as its reasoning engine, orchestrated with
LangGraph.
"""

from __future__ import annotations

import json
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from typing import Annotated, TypedDict

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode

# --------------------------------------------------------------------------- #
# Standard error messages (Policies / Behaviors B4-B7)
# --------------------------------------------------------------------------- #
ERR_EMPTY = "Error: The user request cannot be empty."
ERR_MULTI = (
    "Error: The query spans multiple distinct domains. "
    "Please make one specific research request at a time."
)
ERR_SCOPE = "Error: Query outside of academic scope."
ERR_GENERIC = "Error: Unable to get response from agent."

MODEL_NAME = "gpt-4o"


# --------------------------------------------------------------------------- #
# Tools
# --------------------------------------------------------------------------- #
@tool
def search_arxiv(query: str) -> str:
    """Search for academic papers on arXiv for the Exact Sciences/Technology
    domain (e.g. artificial intelligence, physics, mathematics, engineering).

    Args:
        query: The topic to search papers about.

    Returns:
        Raw JSON string with a list of papers, each having a title and summary.
    """
    try:
        params = urllib.parse.urlencode(
            {
                "search_query": f"all:{query}",
                "start": 0,
                "max_results": 5,
            }
        )
        url = f"http://export.arxiv.org/api/query?{params}"
        with urllib.request.urlopen(url, timeout=30) as resp:
            data = resp.read().decode("utf-8")

        ns = {"atom": "http://www.w3.org/2005/Atom"}
        root = ET.fromstring(data)
        papers = []
        for entry in root.findall("atom:entry", ns):
            title = entry.find("atom:title", ns)
            summary = entry.find("atom:summary", ns)
            papers.append(
                {
                    "title": (title.text or "").strip() if title is not None else "",
                    "summary": (summary.text or "").strip()
                    if summary is not None
                    else "",
                }
            )
        return json.dumps({"source": "arxiv", "query": query, "papers": papers})
    except Exception as exc:  # pragma: no cover - network dependent
        return json.dumps({"source": "arxiv", "query": query, "error": str(exc)})


@tool
def search_pubmed(query: str) -> str:
    """Search for academic papers on PubMed for the Life Sciences/Health domain
    (e.g. medicine, biology, melanoma, genetics).

    Args:
        query: The topic to search papers about.

    Returns:
        Raw JSON string with a list of papers, each having a title and summary.
    """
    try:
        search_params = urllib.parse.urlencode(
            {
                "db": "pubmed",
                "term": query,
                "retmax": 5,
                "retmode": "json",
            }
        )
        search_url = (
            "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi?"
            + search_params
        )
        with urllib.request.urlopen(search_url, timeout=30) as resp:
            ids = json.loads(resp.read().decode("utf-8"))["esearchresult"]["idlist"]

        if not ids:
            return json.dumps({"source": "pubmed", "query": query, "papers": []})

        summary_params = urllib.parse.urlencode(
            {"db": "pubmed", "id": ",".join(ids), "retmode": "json"}
        )
        summary_url = (
            "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esummary.fcgi?"
            + summary_params
        )
        with urllib.request.urlopen(summary_url, timeout=30) as resp:
            result = json.loads(resp.read().decode("utf-8"))["result"]

        papers = []
        for uid in ids:
            item = result.get(uid, {})
            source = item.get("source", "")
            date = item.get("pubdate", "")
            papers.append(
                {
                    "title": item.get("title", ""),
                    "summary": f"Published in {source} ({date}).".strip(),
                }
            )
        return json.dumps({"source": "pubmed", "query": query, "papers": papers})
    except Exception as exc:  # pragma: no cover - network dependent
        return json.dumps({"source": "pubmed", "query": query, "error": str(exc)})


@tool
def get_wikipedia_summary(topic: str) -> str:
    """Retrieve a summary of a single concept from Wikipedia, for the General
    Concepts domain (explaining or defining an established subject).

    Args:
        topic: The concept to explain or define.

    Returns:
        Raw JSON string with the summary text of the concept.
    """
    try:
        title = urllib.parse.quote(topic.replace(" ", "_"))
        url = f"https://en.wikipedia.org/api/rest_v1/page/summary/{title}"
        req = urllib.request.Request(url, headers={"User-Agent": "AcademicTriageAgent/1.0"})
        with urllib.request.urlopen(req, timeout=30) as resp:
            data = json.loads(resp.read().decode("utf-8"))
        return json.dumps(
            {
                "source": "wikipedia",
                "topic": topic,
                "title": data.get("title", topic),
                "summary": data.get("extract", ""),
            }
        )
    except Exception as exc:  # pragma: no cover - network dependent
        return json.dumps({"source": "wikipedia", "topic": topic, "error": str(exc)})


TOOLS = [search_arxiv, search_pubmed, get_wikipedia_summary]
TOOLS_BY_NAME = {t.name: t for t in TOOLS}


# --------------------------------------------------------------------------- #
# LangGraph state and nodes
# --------------------------------------------------------------------------- #
class AgentState(TypedDict):
    messages: Annotated[list, add_messages]


SYSTEM_PROMPT = """You are an academic triage agent. You route a user's \
academic request to exactly ONE tool based on the subject domain, then \
synthesize the tool's results into a natural-language answer.

Available tools and when to use them:
- search_arxiv: ONLY when the user explicitly asks for PAPERS or RESEARCH in \
the Exact Sciences/Technology domain (e.g. artificial intelligence, physics, \
mathematics, computer science, engineering).
- search_pubmed: ONLY when the user explicitly asks for PAPERS or RESEARCH in \
the Life Sciences/Health domain (e.g. medicine, biology, melanoma, genetics).
- get_wikipedia_summary: when the user asks to EXPLAIN or DEFINE a single \
concept (from any academic discipline), when they do NOT explicitly ask for \
papers or research.

Rules:
1. Academic scope is limited to requests for papers, research, or factual \
explanations of established subjects in a recognized academic discipline \
(humanities, arts, social sciences, exact sciences/technology, life \
sciences/health). Personal advice, transactions, creative-generation \
requests, and other non-academic requests are OUT OF SCOPE.
2. If the request is outside the academic scope, respond with EXACTLY this \
text and nothing else: OUT_OF_SCOPE
3. If the request asks about MORE THAN ONE distinct domain in the same text \
(e.g. papers about artificial intelligence AND melanoma), respond with \
EXACTLY this text and nothing else: MULTI_DOMAIN
4. Otherwise, you MUST call exactly ONE tool, exactly ONCE. Never call more \
than one tool.
5. After receiving the tool results, produce the final answer based \
EXCLUSIVELY on the data the tool returned. Do not invent papers, titles, or \
information. For paper searches, write a brief introduction followed by a list \
of the papers, each with its title and a short summary. For concept \
explanations, write one coherent explanatory paragraph.
6. If the tool returned an error or no results, respond with EXACTLY this \
text and nothing else: TOOL_FAILURE"""


def _build_graph():
    llm = ChatOpenAI(model=MODEL_NAME, temperature=0)
    llm_with_tools = llm.bind_tools(TOOLS)

    def triage(state: AgentState) -> AgentState:
        response = llm_with_tools.invoke(state["messages"])
        return {"messages": [response]}

    def synthesize(state: AgentState) -> AgentState:
        # Force a text-only answer after tool execution (no further tool calls).
        response = llm.invoke(state["messages"])
        return {"messages": [response]}

    tool_node = ToolNode(TOOLS)

    def route(state: AgentState):
        last = state["messages"][-1]
        if getattr(last, "tool_calls", None):
            return "tools"
        return END

    graph = StateGraph(AgentState)
    graph.add_node("triage", triage)
    graph.add_node("tools", tool_node)
    graph.add_node("synthesize", synthesize)
    graph.add_edge(START, "triage")
    graph.add_conditional_edges("triage", route, {"tools": "tools", END: END})
    graph.add_edge("tools", "synthesize")
    graph.add_edge("synthesize", END)
    return graph.compile()


_GRAPH = None


def _get_graph():
    global _GRAPH
    if _GRAPH is None:
        _GRAPH = _build_graph()
    return _GRAPH


# --------------------------------------------------------------------------- #
# Entrypoint
# --------------------------------------------------------------------------- #
def agent(user_request: str) -> str:
    """Triage an academic request and return a formatted string answer."""
    # B4 - empty input.
    if user_request is None or not str(user_request).strip():
        return ERR_EMPTY

    try:
        graph = _get_graph()
        initial = {
            "messages": [
                SystemMessage(content=SYSTEM_PROMPT),
                HumanMessage(content=str(user_request)),
            ]
        }
        result = graph.invoke(initial)
        messages = result["messages"]

        # Detect that at most one tool was used and gather tool results.
        tool_calls = 0
        tool_had_results = True
        for msg in messages:
            if isinstance(msg, AIMessage) and getattr(msg, "tool_calls", None):
                tool_calls += len(msg.tool_calls)
            if isinstance(msg, ToolMessage):
                try:
                    payload = json.loads(msg.content)
                    if payload.get("error") or (
                        "papers" in payload and not payload["papers"]
                    ):
                        tool_had_results = False
                except Exception:
                    pass

        final = messages[-1]
        content = final.content if isinstance(final.content, str) else str(final.content)
        content = content.strip()

        # Sentinel handling from the LLM.
        if content == "OUT_OF_SCOPE":
            return ERR_SCOPE
        if content == "MULTI_DOMAIN":
            return ERR_MULTI
        if content == "TOOL_FAILURE":
            return ERR_GENERIC

        # Guard invariants: exactly one tool must have run for a valid request,
        # and the tool must have returned usable results.
        if tool_calls != 1 or not tool_had_results or not content:
            return ERR_GENERIC

        return content
    except Exception:
        return ERR_GENERIC


if __name__ == "__main__":
    import sys

    req = " ".join(sys.argv[1:]) or "Find me recent papers on artificial intelligence"
    print(agent(req))

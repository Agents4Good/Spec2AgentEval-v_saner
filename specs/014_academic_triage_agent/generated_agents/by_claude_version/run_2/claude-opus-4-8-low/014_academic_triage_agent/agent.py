"""Academic Triage Agent.

Triages a user's academic request, selecting exactly one appropriate tool based on
the subject domain (Exact Sciences/Technology, Life Sciences/Health, or General
Concepts) and returns a formatted summary of the results found, using an LLM as its
reasoning engine.

Interface contract:
    - Model:        GPT-4o
    - Framework:    LangGraph
    - GenAI Tool:   ChatOpenAI
    - Entrypoint:   agent(user_request: str) -> str
"""

import json
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from typing import Optional

from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langgraph.graph import END, START, MessagesState, StateGraph

# --- Standard messages (invariant 1) ------------------------------------------
ERR_EMPTY = "Error: The user request cannot be empty."
ERR_MULTI = (
    "Error: The query spans multiple distinct domains. "
    "Please make one specific research request at a time."
)
ERR_SCOPE = "Error: Query outside of academic scope."
ERR_GENERIC = "Error: Unable to get response from agent."

_HEADERS = {"User-Agent": "academic-triage-agent/1.0"}


# --- Tools --------------------------------------------------------------------
@tool
def search_arxiv(query: str) -> str:
    """Search arXiv for academic papers in the Exact Sciences/Technology domain.

    Args:
        query: The search terms describing the desired papers.

    Returns:
        Raw JSON string with a list of papers (title and summary) found on arXiv.
    """
    url = (
        "http://export.arxiv.org/api/query?"
        + urllib.parse.urlencode(
            {"search_query": f"all:{query}", "start": 0, "max_results": 5}
        )
    )
    req = urllib.request.Request(url, headers=_HEADERS)
    with urllib.request.urlopen(req, timeout=30) as resp:
        data = resp.read().decode("utf-8")

    ns = {"atom": "http://www.w3.org/2005/Atom"}
    root = ET.fromstring(data)
    papers = []
    for entry in root.findall("atom:entry", ns):
        title = entry.findtext("atom:title", default="", namespaces=ns)
        summary = entry.findtext("atom:summary", default="", namespaces=ns)
        papers.append(
            {"title": " ".join(title.split()), "summary": " ".join(summary.split())}
        )
    return json.dumps({"source": "arxiv", "papers": papers})


@tool
def search_pubmed(query: str) -> str:
    """Search PubMed for academic papers in the Life Sciences/Health domain.

    Args:
        query: The search terms describing the desired papers.

    Returns:
        Raw JSON string with a list of papers (title and summary) found on PubMed.
    """
    search_url = (
        "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi?"
        + urllib.parse.urlencode(
            {"db": "pubmed", "term": query, "retmode": "json", "retmax": 5}
        )
    )
    req = urllib.request.Request(search_url, headers=_HEADERS)
    with urllib.request.urlopen(req, timeout=30) as resp:
        ids = json.loads(resp.read().decode("utf-8"))["esearchresult"]["idlist"]

    papers = []
    if ids:
        summary_url = (
            "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esummary.fcgi?"
            + urllib.parse.urlencode(
                {"db": "pubmed", "id": ",".join(ids), "retmode": "json"}
            )
        )
        req = urllib.request.Request(summary_url, headers=_HEADERS)
        with urllib.request.urlopen(req, timeout=30) as resp:
            result = json.loads(resp.read().decode("utf-8")).get("result", {})
        for pid in ids:
            item = result.get(pid, {})
            title = item.get("title", "")
            source = item.get("source", "")
            pubdate = item.get("pubdate", "")
            papers.append(
                {
                    "title": title,
                    "summary": f"Published in {source} ({pubdate}).".strip(),
                }
            )
    return json.dumps({"source": "pubmed", "papers": papers})


@tool
def get_wikipedia_summary(topic: str) -> str:
    """Retrieve a summary of a topic from Wikipedia for General Concepts.

    Args:
        topic: The concept or subject to be explained.

    Returns:
        Raw JSON string with the summary retrieved from Wikipedia.
    """
    url = "https://en.wikipedia.org/api/rest_v1/page/summary/" + urllib.parse.quote(
        topic.strip().replace(" ", "_")
    )
    req = urllib.request.Request(url, headers=_HEADERS)
    with urllib.request.urlopen(req, timeout=30) as resp:
        data = json.loads(resp.read().decode("utf-8"))
    return json.dumps(
        {
            "source": "wikipedia",
            "title": data.get("title", topic),
            "extract": data.get("extract", ""),
        }
    )


_TOOLS = {
    "search_arxiv": search_arxiv,
    "search_pubmed": search_pubmed,
    "get_wikipedia_summary": get_wikipedia_summary,
}


# --- LLM ----------------------------------------------------------------------
def _llm() -> ChatOpenAI:
    return ChatOpenAI(model="gpt-4o", temperature=0)


# --- Graph state --------------------------------------------------------------
# Extend LangGraph's prebuilt MessagesState with the triage fields.
class _State(MessagesState):
    category: str
    query: str
    tool_result: str
    final: str


# --- Graph nodes --------------------------------------------------------------
_CLASSIFY_SYSTEM = """You are a triage classifier for an academic research assistant.

Classify the user's request into exactly one category and, when a tool is needed,
extract a concise search query/topic.

Categories:
- "empty": the request is empty or only whitespace.
- "multi_domain": the request asks for papers/research across more than one distinct
  academic domain in the same text (e.g. papers about artificial intelligence AND melanoma).
- "out_of_scope": the request does not seek papers, research, or a factual explanation
  of an established academic subject. This includes personal advice, transactions,
  creative-generation requests, opinions, and other non-academic requests.
- "arxiv": an explicit request for papers or research in Exact Sciences/Technology
  (e.g. artificial intelligence, physics, mathematics, computer science, engineering).
- "pubmed": an explicit request for papers or research in Life Sciences/Health
  (e.g. medicine, biology, melanoma, genetics).
- "wikipedia": a request to explain or define a SINGLE concept (from any discipline,
  including a scientific concept) where the user does NOT explicitly ask for papers or research.

Return ONLY a raw JSON object: {"category": "<one category>", "query": "<search terms or topic, empty if none>"}"""


def classify_node(state: _State) -> dict:
    user_request = state["messages"][-1].content
    resp = _llm().invoke(
        [
            {"role": "system", "content": _CLASSIFY_SYSTEM},
            {"role": "user", "content": user_request},
        ]
    )
    try:
        text = resp.content.strip()
        if text.startswith("```"):
            text = text.split("```")[1].lstrip("json").strip()
        parsed = json.loads(text)
        category = parsed.get("category", "out_of_scope")
        query = parsed.get("query", "") or user_request
    except Exception:
        category = "error"
        query = user_request
    return {
        "messages": state["messages"],
        "category": category,
        "query": query,
    }


def tool_node(state: dict) -> dict:
    category = state["category"]
    query = state["query"]
    tool_map = {
        "arxiv": "search_arxiv",
        "pubmed": "search_pubmed",
        "wikipedia": "get_wikipedia_summary",
    }
    tool_name = tool_map[category]
    selected = _TOOLS[tool_name]
    # Execute the tool exclusively through its LangChain interface.
    raw = selected.invoke({"query": query} if category != "wikipedia" else {"topic": query})
    return {"tool_result": raw, "category": category}


_SYNTH_PAPERS = """You are an academic research assistant. Using ONLY the raw data below,
write a response with a brief introduction followed by a list of papers, each with its
title and a short summary. Do not invent papers, titles, or information. Base the
response exclusively on the provided data.

RAW DATA:
{data}"""

_SYNTH_CONCEPT = """You are an academic research assistant. Using ONLY the raw data below,
write a single coherent explanatory paragraph about the concept. Do not invent
information. Base the response exclusively on the provided data.

RAW DATA:
{data}"""


def synthesize_node(state: dict) -> dict:
    raw = state.get("tool_result", "")
    category = state["category"]
    try:
        parsed = json.loads(raw)
    except Exception:
        parsed = {}

    if category == "wikipedia":
        if not parsed.get("extract"):
            raise ValueError("no results")
        prompt = _SYNTH_CONCEPT.format(data=raw)
    else:
        if not parsed.get("papers"):
            raise ValueError("no results")
        prompt = _SYNTH_PAPERS.format(data=raw)

    resp = _llm().invoke([{"role": "user", "content": prompt}])
    content = resp.content.strip()
    if not content:
        raise ValueError("empty response")
    return {"messages": state["messages"] + [resp], "final": content}


def route(state: dict) -> str:
    category = state["category"]
    if category in ("arxiv", "pubmed", "wikipedia"):
        return "tool"
    return END


def build_graph():
    graph = StateGraph(MessagesState)
    graph.add_node("classify", classify_node)
    graph.add_node("tool", tool_node)
    graph.add_node("synthesize", synthesize_node)
    graph.add_edge(START, "classify")
    graph.add_conditional_edges("classify", route, {"tool": "tool", END: END})
    graph.add_edge("tool", "synthesize")
    graph.add_edge("synthesize", END)
    return graph.compile()


# MessagesState extended with triage fields.
class _State(MessagesState):
    category: str
    query: str
    tool_result: str
    final: str


def agent(user_request: str) -> str:
    """Triage an academic request and return a formatted string response."""
    if user_request is None or not str(user_request).strip():
        return ERR_EMPTY

    try:
        app = build_graph()
        result = app.invoke({"messages": [{"role": "user", "content": user_request}]})

        category = result.get("category")
        if category == "empty":
            return ERR_EMPTY
        if category == "multi_domain":
            return ERR_MULTI
        if category == "out_of_scope":
            return ERR_SCOPE
        if category not in ("arxiv", "pubmed", "wikipedia"):
            return ERR_GENERIC

        final = result.get("final")
        if not final:
            # Fall back to the last message content.
            messages = result.get("messages", [])
            final = messages[-1].content if messages else None
        if not final:
            return ERR_GENERIC
        return final
    except Exception:
        return ERR_GENERIC


if __name__ == "__main__":
    print(agent("Find me recent papers on artificial intelligence"))

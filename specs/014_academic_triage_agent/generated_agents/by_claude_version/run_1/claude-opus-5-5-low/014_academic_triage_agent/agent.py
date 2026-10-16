"""Academic triage agent (LangGraph + ChatOpenAI).

Classifies an academic request into one domain, invokes exactly one tool
(arXiv, PubMed or Wikipedia) and synthesizes a response from its results.
"""

import json
import xml.etree.ElementTree as ET
from typing import Literal, Optional, TypedDict

import requests
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langgraph.graph import END, StateGraph
from pydantic import BaseModel, Field

MODEL_NAME = "gpt-4o"
HTTP_TIMEOUT = 20
MAX_RESULTS = 5
USER_AGENT = "AcademicTriageAgent/1.0 (research tool)"

ERR_EMPTY = "Error: The user request cannot be empty."
ERR_MULTI = "Error: The query spans multiple distinct domains. Please make one specific research request at a time."
ERR_SCOPE = "Error: Query outside of academic scope."
ERR_AGENT = "Error: Unable to get response from agent."


# ----------------------------------------------------------------- tools ---

@tool
def search_arxiv(query: str) -> str:
    """Search arXiv for papers in Exact Sciences/Technology. Returns raw JSON
    list of papers with title, summary, authors and link."""
    resp = requests.get(
        "http://export.arxiv.org/api/query",
        params={"search_query": f"all:{query}", "start": 0, "max_results": MAX_RESULTS},
        headers={"User-Agent": USER_AGENT},
        timeout=HTTP_TIMEOUT,
    )
    resp.raise_for_status()
    ns = {"a": "http://www.w3.org/2005/Atom"}
    root = ET.fromstring(resp.text)
    papers = []
    for entry in root.findall("a:entry", ns):
        papers.append({
            "title": " ".join((entry.findtext("a:title", "", ns) or "").split()),
            "summary": " ".join((entry.findtext("a:summary", "", ns) or "").split()),
            "authors": [a.findtext("a:name", "", ns) for a in entry.findall("a:author", ns)],
            "link": entry.findtext("a:id", "", ns),
        })
    return json.dumps(papers, ensure_ascii=False)


@tool
def search_pubmed(query: str) -> str:
    """Search PubMed for papers in Life Sciences/Health. Returns raw JSON list
    of papers with title, abstract, journal and link."""
    base = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"
    r = requests.get(
        f"{base}/esearch.fcgi",
        params={"db": "pubmed", "term": query, "retmax": MAX_RESULTS, "retmode": "json"},
        headers={"User-Agent": USER_AGENT},
        timeout=HTTP_TIMEOUT,
    )
    r.raise_for_status()
    ids = r.json().get("esearchresult", {}).get("idlist", [])
    if not ids:
        return "[]"
    r = requests.get(
        f"{base}/efetch.fcgi",
        params={"db": "pubmed", "id": ",".join(ids), "retmode": "xml"},
        headers={"User-Agent": USER_AGENT},
        timeout=HTTP_TIMEOUT,
    )
    r.raise_for_status()
    root = ET.fromstring(r.text)
    papers = []
    for art in root.findall(".//PubmedArticle"):
        pmid = art.findtext(".//PMID", "")
        title = "".join(art.find(".//ArticleTitle").itertext()) if art.find(".//ArticleTitle") is not None else ""
        abstract = " ".join("".join(t.itertext()) for t in art.findall(".//AbstractText"))
        papers.append({
            "title": " ".join(title.split()),
            "abstract": " ".join(abstract.split()),
            "journal": art.findtext(".//Journal/Title", ""),
            "link": f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/",
        })
    return json.dumps(papers, ensure_ascii=False)


@tool
def get_wikipedia_summary(topic: str) -> str:
    """Retrieve a Wikipedia summary for a general concept. Returns raw JSON
    with title and extract."""
    headers = {"User-Agent": USER_AGENT}
    s = requests.get(
        "https://en.wikipedia.org/w/api.php",
        params={"action": "query", "list": "search", "srsearch": topic, "srlimit": 1, "format": "json"},
        headers=headers,
        timeout=HTTP_TIMEOUT,
    )
    s.raise_for_status()
    hits = s.json().get("query", {}).get("search", [])
    if not hits:
        return "{}"
    title = hits[0]["title"]
    r = requests.get(
        f"https://en.wikipedia.org/api/rest_v1/page/summary/{requests.utils.quote(title.replace(' ', '_'))}",
        headers=headers,
        timeout=HTTP_TIMEOUT,
    )
    r.raise_for_status()
    data = r.json()
    return json.dumps({"title": data.get("title", title), "extract": data.get("extract", "")}, ensure_ascii=False)


TOOLS = {t.name: t for t in (search_arxiv, search_pubmed, get_wikipedia_summary)}


# ---------------------------------------------------------------- triage ---

class Triage(BaseModel):
    category: Literal["exact_tech", "life_health", "concept", "multi_domain", "out_of_scope"] = Field(
        description="Classification of the request."
    )
    query: str = Field(default="", description="Concise search query / topic (English) for the tool.")


TRIAGE_PROMPT = """You triage academic requests. Classify the user request into exactly one category:
- exact_tech: explicitly asks for papers/research/articles in Exact Sciences or Technology (math, physics, CS, AI, engineering, statistics...).
- life_health: explicitly asks for papers/research/articles in Life Sciences or Health (medicine, biology, diseases, genetics, neuroscience...).
- concept: asks to explain or define a SINGLE established concept from any recognized academic discipline (humanities, arts, social sciences, exact sciences, life sciences), without explicitly asking for papers/research.
- multi_domain: asks about more than one distinct domain/topic in the same request (e.g. papers on AI and melanoma, or explaining two unrelated concepts).
- out_of_scope: not seeking academic information: personal advice, transactions/purchases, creative writing generation, chit-chat, tasks, opinions, etc.
Also give `query`: a concise English search query (for papers) or topic name (for concepts). Leave empty for multi_domain/out_of_scope."""

SYNTH_PAPERS = """You write a response based EXCLUSIVELY on the raw paper data provided. Never invent papers, titles or facts.
Format: a brief introduction sentence, then a list of papers; each item has the exact title and a short summary (1-2 sentences) derived from the provided data. Answer in the user's language."""

SYNTH_CONCEPT = """You write a single coherent explanatory paragraph about the concept based EXCLUSIVELY on the provided Wikipedia summary data. Do not add information that is not in the data. Answer in the user's language."""

ROUTE = {"exact_tech": "search_arxiv", "life_health": "search_pubmed", "concept": "get_wikipedia_summary"}


class State(TypedDict, total=False):
    request: str
    category: str
    query: str
    tool_name: str
    tool_output: Optional[str]
    response: str


def _llm() -> ChatOpenAI:
    return ChatOpenAI(model=MODEL_NAME, temperature=0)


def triage_node(state: State) -> State:
    result = _llm().with_structured_output(Triage).invoke(
        [SystemMessage(content=TRIAGE_PROMPT), HumanMessage(content=state["request"])]
    )
    out: State = {"category": result.category, "query": (result.query or "").strip()}
    if result.category == "multi_domain":
        out["response"] = ERR_MULTI
    elif result.category == "out_of_scope":
        out["response"] = ERR_SCOPE
    else:
        out["tool_name"] = ROUTE[result.category]
        if not out["query"]:
            out["query"] = state["request"]
    return out


def tool_node(state: State) -> State:
    name = state["tool_name"]
    arg = "topic" if name == "get_wikipedia_summary" else "query"
    raw = TOOLS[name].invoke({arg: state["query"]})
    data = json.loads(raw)
    if not data or (isinstance(data, dict) and not data.get("extract")):
        return {"tool_output": None, "response": ERR_AGENT}
    return {"tool_output": raw}


def synth_node(state: State) -> State:
    system = SYNTH_CONCEPT if state["tool_name"] == "get_wikipedia_summary" else SYNTH_PAPERS
    msg = _llm().invoke([
        SystemMessage(content=system),
        HumanMessage(content=f"User request: {state['request']}\n\nRaw tool data:\n{state['tool_output']}"),
    ])
    text = (msg.content or "").strip() if isinstance(msg.content, str) else ""
    return {"response": text or ERR_AGENT}


def build_graph():
    g = StateGraph(State)
    g.add_node("triage", triage_node)
    g.add_node("tool", tool_node)
    g.add_node("synthesize", synth_node)
    g.set_entry_point("triage")
    g.add_conditional_edges("triage", lambda s: END if s.get("response") else "tool", {"tool": "tool", END: END})
    g.add_conditional_edges("tool", lambda s: END if s.get("response") else "synthesize", {"synthesize": "synthesize", END: END})
    g.add_edge("synthesize", END)
    return g.compile()


_GRAPH = None


def agent(user_request: str) -> str:
    if not isinstance(user_request, str) or not user_request.strip():
        return ERR_EMPTY
    global _GRAPH
    try:
        if _GRAPH is None:
            _GRAPH = build_graph()
        result = _GRAPH.invoke({"request": user_request.strip()})
        response = result.get("response")
        return response if isinstance(response, str) and response.strip() else ERR_AGENT
    except Exception:
        return ERR_AGENT


if __name__ == "__main__":
    import sys
    print(agent(" ".join(sys.argv[1:])))

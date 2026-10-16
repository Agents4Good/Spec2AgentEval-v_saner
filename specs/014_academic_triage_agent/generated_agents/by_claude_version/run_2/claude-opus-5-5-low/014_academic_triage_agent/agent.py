"""Academic triage agent: routes an academic request to exactly one tool
(arXiv, PubMed or Wikipedia) and synthesizes the results with GPT-4o."""

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
USER_AGENT = "AcademicTriageAgent/1.0 (educational use)"

ERR_EMPTY = "Error: The user request cannot be empty."
ERR_MULTI = (
    "Error: The query spans multiple distinct domains. "
    "Please make one specific research request at a time."
)
ERR_SCOPE = "Error: Query outside of academic scope."
ERR_FAIL = "Error: Unable to get response from agent."


# ---------------------------------------------------------------- tools
@tool
def search_arxiv(query: str) -> str:
    """Search arXiv for papers (Exact Sciences/Technology). Returns raw JSON
    list of papers with title, summary, authors, published date and link."""
    resp = requests.get(
        "http://export.arxiv.org/api/query",
        params={"search_query": f"all:{query}", "start": 0,
                "max_results": MAX_RESULTS, "sortBy": "relevance"},
        headers={"User-Agent": USER_AGENT},
        timeout=HTTP_TIMEOUT,
    )
    resp.raise_for_status()
    ns = {"a": "http://www.w3.org/2005/Atom"}
    root = ET.fromstring(resp.content)
    papers = []
    for entry in root.findall("a:entry", ns):
        papers.append({
            "title": " ".join((entry.findtext("a:title", "", ns)).split()),
            "summary": " ".join((entry.findtext("a:summary", "", ns)).split()),
            "authors": [a.findtext("a:name", "", ns)
                        for a in entry.findall("a:author", ns)],
            "published": entry.findtext("a:published", "", ns),
            "link": entry.findtext("a:id", "", ns),
        })
    return json.dumps(papers, ensure_ascii=False)


@tool
def search_pubmed(query: str) -> str:
    """Search PubMed for papers (Life Sciences/Health). Returns raw JSON
    list of papers with title, abstract, journal, date and link."""
    base = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"
    headers = {"User-Agent": USER_AGENT}
    search = requests.get(
        f"{base}/esearch.fcgi",
        params={"db": "pubmed", "term": query, "retmax": MAX_RESULTS,
                "retmode": "json", "sort": "relevance"},
        headers=headers, timeout=HTTP_TIMEOUT,
    )
    search.raise_for_status()
    ids = search.json().get("esearchresult", {}).get("idlist", [])
    if not ids:
        return "[]"
    fetch = requests.get(
        f"{base}/efetch.fcgi",
        params={"db": "pubmed", "id": ",".join(ids), "retmode": "xml"},
        headers=headers, timeout=HTTP_TIMEOUT,
    )
    fetch.raise_for_status()
    root = ET.fromstring(fetch.content)
    papers = []
    for art in root.findall(".//PubmedArticle"):
        pmid = art.findtext(".//PMID", "")
        title = "".join(art.find(".//ArticleTitle").itertext()) \
            if art.find(".//ArticleTitle") is not None else ""
        abstract = " ".join("".join(t.itertext())
                            for t in art.findall(".//Abstract/AbstractText"))
        papers.append({
            "title": " ".join(title.split()),
            "abstract": " ".join(abstract.split()),
            "journal": art.findtext(".//Journal/Title", ""),
            "year": art.findtext(".//PubDate/Year", ""),
            "link": f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/",
        })
    return json.dumps(papers, ensure_ascii=False)


@tool
def get_wikipedia_summary(topic: str) -> str:
    """Retrieve a Wikipedia summary for a topic (General Concepts). Returns
    raw JSON with title, extract and link, or an empty object if not found."""
    headers = {"User-Agent": USER_AGENT}
    found = requests.get(
        "https://en.wikipedia.org/w/api.php",
        params={"action": "opensearch", "search": topic, "limit": 1,
                "namespace": 0, "format": "json"},
        headers=headers, timeout=HTTP_TIMEOUT,
    )
    found.raise_for_status()
    titles = found.json()[1]
    if not titles:
        return "{}"
    title = titles[0].replace(" ", "_")
    resp = requests.get(
        f"https://en.wikipedia.org/api/rest_v1/page/summary/{title}",
        headers=headers, timeout=HTTP_TIMEOUT,
    )
    if resp.status_code == 404:
        return "{}"
    resp.raise_for_status()
    data = resp.json()
    return json.dumps({
        "title": data.get("title", ""),
        "extract": data.get("extract", ""),
        "link": data.get("content_urls", {}).get("desktop", {}).get("page", ""),
    }, ensure_ascii=False)


TOOLS = {t.name: t for t in (search_arxiv, search_pubmed, get_wikipedia_summary)}


# ---------------------------------------------------------------- LLM schema
class Triage(BaseModel):
    category: Literal["exact_tech", "life_health", "concept",
                      "multi_domain", "out_of_scope"] = Field(
        description="Classification of the request.")
    query: str = Field(default="",
                       description="Concise English search query/topic for the tool.")


TRIAGE_PROMPT = """You triage academic requests. Classify the user's request into exactly one category:
- exact_tech: explicitly asks for academic papers or research in Exact Sciences/Technology (math, physics, computer science, AI, engineering, statistics, etc.).
- life_health: explicitly asks for academic papers or research in Life Sciences/Health (medicine, biology, diseases, genetics, nursing, pharmacology, etc.).
- concept: asks to explain or define a SINGLE concept from any recognized academic discipline (humanities, arts, social sciences, exact or life sciences) without explicitly asking for papers or research.
- multi_domain: asks about more than one distinct domain/subject in the same request (e.g., papers on AI and on melanoma; or explain two unrelated concepts; or papers in one domain plus a concept in another).
- out_of_scope: does not seek papers, research, or a factual explanation of an established academic subject (personal advice, transactions, shopping, creative writing, jokes, chit-chat, coding help, etc.).
For exact_tech/life_health/concept, set `query` to a concise English search query (for concept: just the concept name). Otherwise leave it empty."""

SYNTH_PAPERS_PROMPT = """You write a response using ONLY the raw paper data provided. Never invent papers, titles, or facts.
Format: a brief introduction sentence, then a numbered list where each item has the paper title (bold) followed by a 1-2 sentence summary based only on the provided data. Answer in the user's language."""

SYNTH_CONCEPT_PROMPT = """You write one coherent explanatory paragraph about the concept using ONLY the raw Wikipedia summary provided. Do not add information that is not in it. Answer in the user's language."""


# ---------------------------------------------------------------- graph
class State(TypedDict, total=False):
    request: str
    category: str
    query: str
    tool_name: Optional[str]
    tool_output: Optional[str]
    response: str


CATEGORY_TO_TOOL = {
    "exact_tech": "search_arxiv",
    "life_health": "search_pubmed",
    "concept": "get_wikipedia_summary",
}


def _llm() -> ChatOpenAI:
    return ChatOpenAI(model=MODEL_NAME, temperature=0)


def triage_node(state: State) -> State:
    result: Triage = _llm().with_structured_output(Triage).invoke([
        SystemMessage(content=TRIAGE_PROMPT),
        HumanMessage(content=state["request"]),
    ])
    if result.category == "multi_domain":
        return {"category": result.category, "response": ERR_MULTI}
    if result.category == "out_of_scope":
        return {"category": result.category, "response": ERR_SCOPE}
    query = result.query.strip() or state["request"]
    return {"category": result.category, "query": query,
            "tool_name": CATEGORY_TO_TOOL[result.category]}


def tool_node(state: State) -> State:
    name = state["tool_name"]
    arg = "topic" if name == "get_wikipedia_summary" else "query"
    output = TOOLS[name].invoke({arg: state["query"]})  # exactly one call
    return {"tool_output": output}


def synth_node(state: State) -> State:
    raw = state.get("tool_output") or ""
    try:
        data = json.loads(raw)
    except (TypeError, ValueError):
        data = None
    if not data or (isinstance(data, dict) and not data.get("extract")):
        return {"response": ERR_FAIL}
    prompt = (SYNTH_CONCEPT_PROMPT if state["tool_name"] == "get_wikipedia_summary"
              else SYNTH_PAPERS_PROMPT)
    msg = _llm().invoke([
        SystemMessage(content=prompt),
        HumanMessage(content=f"User request: {state['request']}\n\nRaw data:\n{raw}"),
    ])
    text = (msg.content or "").strip() if isinstance(msg.content, str) else ""
    return {"response": text or ERR_FAIL}


def _route(state: State) -> str:
    return "tool" if state.get("tool_name") else END


def build_graph():
    g = StateGraph(State)
    g.add_node("triage", triage_node)
    g.add_node("tool", tool_node)
    g.add_node("synthesize", synth_node)
    g.set_entry_point("triage")
    g.add_conditional_edges("triage", _route, {"tool": "tool", END: END})
    g.add_edge("tool", "synthesize")
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
        return response if isinstance(response, str) and response else ERR_FAIL
    except Exception:
        return ERR_FAIL


if __name__ == "__main__":
    import sys
    print(agent(" ".join(sys.argv[1:]) or input("Request: ")))

"""Chef agent: generates personalized recipes (LangGraph + ChatOpenAI GPT-4o)."""
from __future__ import annotations

import json
import os
from typing import Any, List, Optional, TypedDict

import requests
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langgraph.graph import END, StateGraph
from pydantic import BaseModel, Field

TIMEOUT = 15
MODEL = "gpt-4o"


# --------------------------------------------------------------------------- tools
def _extract_block_text(block: dict) -> str:
    data = block.get(block.get("type", ""), {}) or {}
    rich = data.get("rich_text") or data.get("text") or []
    return "".join(r.get("plain_text", "") for r in rich if isinstance(r, dict)).strip()


@tool
def retrieve_pantry() -> Any:
    """Retrieve the complete list of ingredients available in the user's pantry (Notion page)."""
    api_key = os.getenv("NOTION_API_KEY")
    page_id = os.getenv("NOTION_PAGE_ID")
    if not api_key or not page_id:
        return "Error: missing Notion credentials (NOTION_API_KEY / NOTION_PAGE_ID)."
    headers = {"Authorization": f"Bearer {api_key}", "Notion-Version": "2022-06-28"}
    url = f"https://api.notion.com/v1/blocks/{page_id}/children?page_size=100"
    items: List[str] = []
    try:
        cursor = None
        while True:
            full = url + (f"&start_cursor={cursor}" if cursor else "")
            resp = requests.get(full, headers=headers, timeout=TIMEOUT)
            if resp.status_code != 200:
                return f"Error: Notion page inaccessible (HTTP {resp.status_code})."
            payload = resp.json()
            for block in payload.get("results", []):
                text = _extract_block_text(block)
                if not text:
                    continue
                for part in text.replace(";", "\n").split("\n"):
                    part = part.strip(" -•*\t")
                    if part:
                        items.append(part)
            if not payload.get("has_more"):
                break
            cursor = payload.get("next_cursor")
    except requests.Timeout:
        return "Error: timeout while accessing the Notion page."
    except Exception as exc:  # noqa: BLE001
        return f"Error: failed to access the Notion page: {exc}"
    return items


@tool
def search_recipes(ingredients: List[str]) -> Any:
    """Search the web (Tavily) for reference recipes using the given ingredients. Returns up to 5 results."""
    api_key = os.getenv("TAVILY_API_KEY")
    if not api_key:
        return "erro: missing Tavily credentials (TAVILY_API_KEY)."
    query = "recipe with " + ", ".join(ingredients)
    try:
        resp = requests.post(
            "https://api.tavily.com/search",
            json={"api_key": api_key, "query": query, "max_results": 5},
            headers={"Authorization": f"Bearer {api_key}"},
            timeout=TIMEOUT,
        )
        if resp.status_code != 200:
            return f"erro: Tavily inaccessible (HTTP {resp.status_code})."
        results = resp.json().get("results", []) or []
    except requests.Timeout:
        return "erro: timeout while searching recipes on Tavily."
    except Exception as exc:  # noqa: BLE001
        return f"erro: failed to search recipes: {exc}"
    return [
        {"title": r.get("title", ""), "url": r.get("url", ""), "content": r.get("content", "")}
        for r in results[:5]
    ]


# --------------------------------------------------------------------------- schemas
class Validation(BaseModel):
    is_valid: bool = Field(description="True if the query is a non-empty, non-offensive food/cooking request.")
    reason: str = Field(description="Short explanation.")


class Ingredients(BaseModel):
    dietary_restrictions: List[str] = Field(default_factory=list)
    ingredients: List[str] = Field(default_factory=list)


class FilteredIngredients(BaseModel):
    ingredients: List[str] = Field(default_factory=list)


class Recipe(BaseModel):
    title: str
    ingredients: List[str] = Field(description="Each item with quantity, e.g. '200 g rice'.")
    equipment: List[str]
    instructions: List[str]
    storage: str


class AgentState(TypedDict, total=False):
    query: str
    is_valid: bool
    dietary_restrictions: List[str]
    ingredients: List[str]
    pantry_ingredients: List[str]
    reference_recipes: List[dict]
    recipe: Optional[dict]
    tool_calls: List[str]
    error: Optional[str]
    status: str


PERSONA = "You are a professional chef who creates personalized recipes."

DIET_RULES = (
    "A dietary restriction implies ALL of its derived ingredients even if not mentioned "
    "(e.g. vegan excludes meat, fish, eggs, dairy, honey, butter, cheese, gelatin; vegetarian excludes "
    "meat, fish, seafood, gelatin; lactose-free excludes milk, cheese, butter, cream, yogurt; "
    "gluten-free excludes wheat, flour, bread, pasta, barley, rye; allergies exclude the allergen and its derivatives)."
)


def _llm() -> ChatOpenAI:
    return ChatOpenAI(model=MODEL, temperature=0, timeout=60)


def _fail(state: AgentState, msg: str) -> AgentState:
    msg = msg if msg.startswith("Error:") else f"Error: {msg}"
    return {"error": msg, "recipe": None, "status": "error"}


# --------------------------------------------------------------------------- nodes
def validate_node(state: AgentState) -> AgentState:
    query = (state.get("query") or "").strip()
    if not query:
        return {"is_valid": False, **_fail(state, "Error: invalid query - the query is empty.")}
    try:
        res: Validation = _llm().with_structured_output(Validation).invoke([
            SystemMessage(content=(
                f"{PERSONA} Classify the user query. It is INVALID if it is empty, offensive, or unrelated "
                "to food/cooking/recipes. Otherwise it is VALID (e.g. any request for a dish, meal, or recipe, "
                "even without ingredients)."
            )),
            HumanMessage(content=query),
        ])
    except Exception as exc:  # noqa: BLE001
        return {"is_valid": False, **_fail(state, f"model failure during validation: {exc}")}
    if not res.is_valid:
        return {"is_valid": False, **_fail(state, f"Error: invalid query - {res.reason}")}
    return {"is_valid": True}


def identify_node(state: AgentState) -> AgentState:
    try:
        res: Ingredients = _llm().with_structured_output(Ingredients).invoke([
            SystemMessage(content=(
                f"{PERSONA} Extract from the query: (1) the declared dietary restrictions; (2) the ingredients "
                "the user explicitly HAS / wants to use. Rules: exclude ingredients mentioned under negation "
                "(\"I don't have eggs\", \"without milk\"), absence (\"I ran out of\") or hypothesis (\"if I had\", "
                "\"maybe\"). Normalize regional variants and inflected forms to a singular canonical name. "
                f"{DIET_RULES} Remove any ingredient incompatible with the restrictions. Generic dish names "
                "or meal types are NOT ingredients. Return an empty list if none remain."
            )),
            HumanMessage(content=state["query"]),
        ])
    except Exception as exc:  # noqa: BLE001
        return _fail(state, f"model failure during ingredient identification: {exc}")
    seen, ings = set(), []
    for i in res.ingredients:
        k = i.strip().lower()
        if k and k not in seen:
            seen.add(k)
            ings.append(i.strip())
    return {"ingredients": ings, "dietary_restrictions": res.dietary_restrictions}


def search_node(state: AgentState) -> AgentState:
    calls = list(state.get("tool_calls", [])) + ["search_recipes"]
    try:
        out = search_recipes.invoke({"ingredients": state["ingredients"]})
    except Exception as exc:  # noqa: BLE001
        return {"tool_calls": calls, **_fail(state, f"search_recipes failed: {exc}")}
    if isinstance(out, str):
        detail = out.split(":", 1)[1].strip() if ":" in out else out
        return {"tool_calls": calls, **_fail(state, f"execution failure in search_recipes: {detail}")}
    return {"tool_calls": calls, "reference_recipes": out or []}


def pantry_node(state: AgentState) -> AgentState:
    calls = list(state.get("tool_calls", [])) + ["retrieve_pantry"]
    try:
        out = retrieve_pantry.invoke({})
    except Exception as exc:  # noqa: BLE001
        return {"tool_calls": calls, **_fail(state, f"retrieve_pantry failed: {exc}")}
    if isinstance(out, str):
        detail = out.split(":", 1)[1].strip() if ":" in out else out
        return {"tool_calls": calls, **_fail(state, f"execution failure in retrieve_pantry: {detail}")}
    if not out:
        return {"tool_calls": calls, "pantry_ingredients": []}
    try:
        res: FilteredIngredients = _llm().with_structured_output(FilteredIngredients).invoke([
            SystemMessage(content=(
                f"{PERSONA} Given the user query and the pantry list, return only the pantry items compatible "
                f"with the dietary restrictions declared in the query. {DIET_RULES} If no restriction is "
                "declared, return all items. Keep item names exactly as given."
            )),
            HumanMessage(content=f"Query: {state['query']}\nPantry: {json.dumps(out, ensure_ascii=False)}"),
        ])
    except Exception as exc:  # noqa: BLE001
        return {"tool_calls": calls, **_fail(state, f"model failure while filtering pantry: {exc}")}
    allowed = {o.lower(): o for o in out}
    filtered = [allowed[i.lower()] for i in res.ingredients if i.lower() in allowed]
    return {"tool_calls": calls, "pantry_ingredients": filtered}


def lack_node(state: AgentState) -> AgentState:
    return _fail(state, "Error: no ingredients compatible with the dietary restrictions are available in the pantry.")


def generate_node(state: AgentState) -> AgentState:
    base = state.get("ingredients") or state.get("pantry_ingredients") or []
    refs = state.get("reference_recipes") or []
    try:
        res: Recipe = _llm().with_structured_output(Recipe).invoke([
            SystemMessage(content=(
                f"{PERSONA} Create one complete recipe that respects the user's preferences and dietary "
                f"restrictions. {DIET_RULES} The recipe MUST use at least one of the available ingredients "
                "(use their names as given) and must not include any ingredient violating the restrictions. "
                "Reference recipes are only preparation inspiration, not a source of base ingredients. "
                "Every field must be non-empty; ingredients include quantities."
            )),
            HumanMessage(content=(
                f"Query: {state['query']}\nAvailable ingredients: {json.dumps(base, ensure_ascii=False)}\n"
                f"Reference recipes: {json.dumps(refs, ensure_ascii=False)[:6000]}"
            )),
        ])
    except Exception as exc:  # noqa: BLE001
        return _fail(state, f"model failure during recipe generation: {exc}")
    recipe = res.model_dump()
    if not all(recipe.get(k) for k in ("title", "ingredients", "equipment", "instructions", "storage")):
        return _fail(state, "generated recipe is incomplete.")
    return {"recipe": recipe, "status": "success", "error": None}


# --------------------------------------------------------------------------- graph
def _after_validate(s: AgentState) -> str:
    return "identify" if s.get("is_valid") else END


def _after_identify(s: AgentState) -> str:
    if s.get("error"):
        return END
    return "search" if s.get("ingredients") else "pantry"


def _after_search(s: AgentState) -> str:
    return END if s.get("error") else "generate"


def _after_pantry(s: AgentState) -> str:
    if s.get("error"):
        return END
    return "generate" if s.get("pantry_ingredients") else "lack"


def build_graph():
    g = StateGraph(AgentState)
    for name, fn in [("validate", validate_node), ("identify", identify_node), ("search", search_node),
                     ("pantry", pantry_node), ("lack", lack_node), ("generate", generate_node)]:
        g.add_node(name, fn)
    g.set_entry_point("validate")
    g.add_conditional_edges("validate", _after_validate, ["identify", END])
    g.add_conditional_edges("identify", _after_identify, ["search", "pantry", END])
    g.add_conditional_edges("search", _after_search, ["generate", END])
    g.add_conditional_edges("pantry", _after_pantry, ["generate", "lack", END])
    g.add_edge("lack", END)
    g.add_edge("generate", END)
    return g.compile()


def agent(query: str) -> dict:
    """Entrypoint: run the chef agent and return the final state dict."""
    init: AgentState = {
        "query": query or "", "is_valid": False, "dietary_restrictions": [], "ingredients": [],
        "pantry_ingredients": [], "reference_recipes": [], "recipe": None, "tool_calls": [],
        "error": None, "status": "running",
    }
    try:
        return dict(build_graph().invoke(init))
    except Exception as exc:  # noqa: BLE001
        return {**init, "error": f"Error: unexpected execution failure: {exc}", "status": "error"}


if __name__ == "__main__":
    import sys
    print(json.dumps(agent(" ".join(sys.argv[1:])), indent=2, ensure_ascii=False))

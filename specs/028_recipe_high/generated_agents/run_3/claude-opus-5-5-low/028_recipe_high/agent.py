"""Personalized recipe generator agent (chef persona) built with LangGraph + ChatOpenAI."""
import json
import os
from typing import List, Optional, TypedDict

import requests
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langgraph.graph import END, StateGraph
from pydantic import BaseModel, Field

TIMEOUT = 20
NOTION_VERSION = "2022-06-28"


# --------------------------------------------------------------------------- tools
def _rich_text(block: dict) -> str:
    data = block.get(block.get("type", ""), {}) or {}
    return "".join(rt.get("plain_text", "") for rt in data.get("rich_text", []) or []).strip()


@tool
def retrieve_pantry() -> str:
    """Retrieve the list of ingredients available in the user's pantry from a Notion page.
    Returns a JSON list of ingredient names, or a message starting with 'Error:'."""
    api_key = os.getenv("NOTION_API_KEY")
    page_id = os.getenv("NOTION_PAGE_ID")
    if not api_key or not page_id:
        return "Error: missing Notion credentials (NOTION_API_KEY / NOTION_PAGE_ID)."
    headers = {"Authorization": f"Bearer {api_key}", "Notion-Version": NOTION_VERSION}
    items: List[str] = []
    cursor = None
    try:
        while True:
            params = {"page_size": 100}
            if cursor:
                params["start_cursor"] = cursor
            resp = requests.get(
                f"https://api.notion.com/v1/blocks/{page_id}/children",
                headers=headers, params=params, timeout=TIMEOUT,
            )
            if resp.status_code != 200:
                return f"Error: Notion page inaccessible (HTTP {resp.status_code}): {resp.text[:200]}"
            data = resp.json()
            for block in data.get("results", []):
                if block.get("type") in ("heading_1", "heading_2", "heading_3"):
                    continue
                text = _rich_text(block)
                if not text:
                    continue
                for part in text.replace(";", ",").split(","):
                    part = part.strip(" -•*\t")
                    if part:
                        items.append(part)
            if not data.get("has_more"):
                break
            cursor = data.get("next_cursor")
    except requests.Timeout:
        return "Error: timeout while accessing the Notion page."
    except Exception as exc:  # noqa: BLE001
        return f"Error: failed to retrieve pantry: {exc}"
    return json.dumps(items, ensure_ascii=False)


@tool
def search_recipes(ingredients: List[str]) -> str:
    """Search the web (Tavily) for reference recipes using the given compatible ingredients.
    Returns a JSON list (max 5) of {title, url, content}, or a message starting with 'Error:'."""
    api_key = os.getenv("TAVILY_API_KEY")
    if not api_key:
        return "Error: missing Tavily credentials (TAVILY_API_KEY)."
    query = "recipe with " + ", ".join(ingredients)
    try:
        resp = requests.post(
            "https://api.tavily.com/search",
            json={"api_key": api_key, "query": query, "max_results": 5},
            headers={"Authorization": f"Bearer {api_key}"},
            timeout=TIMEOUT,
        )
        if resp.status_code != 200:
            return f"Error: Tavily search failed (HTTP {resp.status_code}): {resp.text[:200]}"
        results = resp.json().get("results", []) or []
    except requests.Timeout:
        return "Error: timeout while searching recipes on Tavily."
    except Exception as exc:  # noqa: BLE001
        return f"Error: failed to search recipes: {exc}"
    out = [
        {"title": r.get("title", ""), "url": r.get("url", ""), "content": (r.get("content") or "")[:800]}
        for r in results[:5]
    ]
    return json.dumps(out, ensure_ascii=False)


# --------------------------------------------------------------------------- LLM schemas
class Validation(BaseModel):
    is_valid: bool = Field(description="True if the query is a non-empty, non-offensive food/cooking request.")
    reason: str = Field(description="Short explanation of the classification.")


class Ingredients(BaseModel):
    restrictions: List[str] = Field(description="Dietary restrictions declared in the query.")
    ingredients: List[str] = Field(description="Usable ingredients compatible with the restrictions.")


class Recipe(BaseModel):
    title: str
    ingredients: List[str] = Field(description="Ingredients with quantities.")
    equipment: List[str]
    instructions: List[str] = Field(description="Ordered step-by-step instructions.")
    storage: str


def _llm() -> ChatOpenAI:
    return ChatOpenAI(model="gpt-4o", temperature=0, timeout=60, max_retries=2)


VALIDATION_PROMPT = """You are a chef's assistant classifying user queries for a recipe generator.
A query is INVALID if it is: empty/meaningless, offensive/abusive, or unrelated to food/cooking/recipes.
Any other query is VALID (including vague requests like "make me something for dinner")."""

IDENTIFY_PROMPT = """You are a chef identifying the ingredients usable in a recipe.
Usable ingredients = ingredients explicitly mentioned in the user query as available/desired
PLUS ALL ingredients from the pantry list. Then keep only those compatible with the declared dietary restrictions.
Rules:
- A dietary restriction implies all derived ingredients, even if not mentioned (e.g. vegan excludes meat, fish,
  eggs, milk, butter, cheese, honey; lactose-free excludes milk, cheese, butter, cream, yogurt; gluten-free excludes
  wheat flour, bread, pasta, barley, rye; vegetarian excludes meat, poultry, fish, seafood, gelatin; allergies exclude
  the allergen and its derivatives).
- An ingredient mentioned under negation, absence or hypothesis ("I don't have eggs", "no milk", "if I had chicken",
  "maybe I'll buy tomatoes") is NOT available and must not be included (unless it is also in the pantry and
  not otherwise excluded... but never include it if the user says they do not have it).
- Treat regional variants and inflected forms as the same ingredient; output each ingredient once, in a canonical
  singular form, in the language of the query.
- Do not invent ingredients. If none remain, return an empty list."""

RECIPE_PROMPT = """You are a creative, experienced chef who creates personalized recipes.
Create ONE complete recipe primarily based on the compatible ingredients provided, honoring the user's preferences.
Reference recipes from the web are only preparation inspiration: they are NOT a source of base ingredients.
Strict rules:
- Use at least one of the compatible ingredients (prefer using several).
- Never include any ingredient that violates the declared dietary restrictions (including derived ingredients).
- You may add only basic staples (water, salt, pepper, common spices, oil) when compatible with the restrictions.
- All fields must be non-empty; list ingredients with quantities.
Answer in the same language as the user query."""


# --------------------------------------------------------------------------- graph
class State(TypedDict, total=False):
    query: str
    error: Optional[str]
    pantry: List[str]
    restrictions: List[str]
    ingredients: List[str]
    references: list
    output: str


def validate_node(state: State) -> State:
    query = (state.get("query") or "").strip()
    if not query:
        return {"error": "Error: invalid query - the query is empty."}
    res: Validation = _llm().with_structured_output(Validation).invoke(
        [SystemMessage(content=VALIDATION_PROMPT), HumanMessage(content=query)]
    )
    if not res.is_valid:
        return {"error": f"Error: invalid query - {res.reason}"}
    return {}


def pantry_node(state: State) -> State:
    raw = retrieve_pantry.invoke({})
    if isinstance(raw, str) and raw.strip().lower().startswith(("error", "erro:")):
        return {"error": raw if raw.startswith("Error:") else f"Error: {raw}"}
    try:
        pantry = json.loads(raw)
    except Exception:  # noqa: BLE001
        return {"error": "Error: could not parse pantry data."}
    return {"pantry": pantry}


def identify_node(state: State) -> State:
    payload = f"User query:\n{state['query']}\n\nPantry ingredients:\n{json.dumps(state.get('pantry', []), ensure_ascii=False)}"
    res: Ingredients = _llm().with_structured_output(Ingredients).invoke(
        [SystemMessage(content=IDENTIFY_PROMPT), HumanMessage(content=payload)]
    )
    ingredients = [i.strip() for i in res.ingredients if i and i.strip()]
    if not ingredients:
        return {
            "restrictions": res.restrictions,
            "ingredients": [],
            "error": "Error: no ingredients compatible with the dietary restrictions were found in the query or pantry.",
        }
    return {"restrictions": res.restrictions, "ingredients": ingredients}


def search_node(state: State) -> State:
    raw = search_recipes.invoke({"ingredients": state["ingredients"]})
    if isinstance(raw, str) and raw.strip().lower().startswith(("error", "erro:")):
        return {"error": raw if raw.startswith("Error:") else f"Error: {raw}"}
    try:
        refs = json.loads(raw)
    except Exception:  # noqa: BLE001
        refs = []
    return {"references": refs}


def _format(r: Recipe) -> str:
    lines = [f"Title: {r.title}", "", "Ingredients:"]
    lines += [f"- {i}" for i in r.ingredients]
    lines += ["", "Equipment:"] + [f"- {e}" for e in r.equipment]
    lines += ["", "Instructions:"] + [f"{n}. {s}" for n, s in enumerate(r.instructions, 1)]
    lines += ["", f"Storage: {r.storage}"]
    return "\n".join(lines)


def recipe_node(state: State) -> State:
    payload = (
        f"User query:\n{state['query']}\n\n"
        f"Declared dietary restrictions: {json.dumps(state.get('restrictions', []), ensure_ascii=False)}\n"
        f"Compatible ingredients: {json.dumps(state['ingredients'], ensure_ascii=False)}\n\n"
        f"Reference recipes (inspiration only):\n{json.dumps(state.get('references', []), ensure_ascii=False)}"
    )
    r: Recipe = _llm().with_structured_output(Recipe).invoke(
        [SystemMessage(content=RECIPE_PROMPT), HumanMessage(content=payload)]
    )
    if not (r.title.strip() and r.ingredients and r.equipment and r.instructions and r.storage.strip()):
        return {"error": "Error: the generated recipe is incomplete."}
    return {"output": _format(r)}


def _route(next_node: str):
    return lambda s: "end" if s.get("error") else next_node


def build_graph():
    g = StateGraph(State)
    g.add_node("validate", validate_node)
    g.add_node("pantry", pantry_node)
    g.add_node("identify", identify_node)
    g.add_node("search", search_node)
    g.add_node("recipe", recipe_node)
    g.set_entry_point("validate")
    g.add_conditional_edges("validate", _route("pantry"), {"pantry": "pantry", "end": END})
    g.add_conditional_edges("pantry", _route("identify"), {"identify": "identify", "end": END})
    g.add_conditional_edges("identify", _route("search"), {"search": "search", "end": END})
    g.add_conditional_edges("search", _route("recipe"), {"recipe": "recipe", "end": END})
    g.add_edge("recipe", END)
    return g.compile()


def agent(query: str) -> str:
    try:
        result = build_graph().invoke({"query": query if isinstance(query, str) else ""})
        if result.get("error"):
            return result["error"]
        return result.get("output") or "Error: no recipe was generated."
    except Exception as exc:  # noqa: BLE001
        return f"Error: execution failure - {exc}"


if __name__ == "__main__":
    import sys
    print(agent(" ".join(sys.argv[1:]) or "Make me a vegan dinner with tomatoes"))

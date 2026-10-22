"""Personalized recipe generator agent (chef persona) built with LangGraph."""

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
    parts = data.get("rich_text") or data.get("text") or []
    return "".join(p.get("plain_text", "") for p in parts if isinstance(p, dict)).strip()


@tool
def retrieve_pantry() -> str:
    """Retrieve the list of ingredients available in the user's pantry from a Notion page.

    Returns a JSON list of ingredient strings (empty list if the pantry is empty),
    or a message starting with 'Error:' on failure."""
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
                text = _rich_text(block)
                if not text:
                    continue
                for line in text.splitlines():
                    for piece in line.split(","):
                        piece = piece.strip(" -•*\t")
                        if piece:
                            items.append(piece)
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

    Returns a JSON list of up to 5 results with title, url and content (empty list if none),
    or a message starting with 'Error:' on failure."""
    api_key = os.getenv("TAVILY_API_KEY")
    if not api_key:
        return "Error: missing Tavily credentials (TAVILY_API_KEY)."
    try:
        resp = requests.post(
            "https://api.tavily.com/search",
            json={"api_key": api_key, "query": "recipe with " + ", ".join(ingredients), "max_results": 5},
            headers={"Authorization": f"Bearer {api_key}"},
            timeout=TIMEOUT,
        )
        if resp.status_code != 200:
            return f"Error: Tavily search failed (HTTP {resp.status_code}): {resp.text[:200]}"
        results = resp.json().get("results", [])[:5]
    except requests.Timeout:
        return "Error: timeout while searching recipes on Tavily."
    except Exception as exc:  # noqa: BLE001
        return f"Error: failed to search recipes: {exc}"
    return json.dumps(
        [{"title": r.get("title", ""), "url": r.get("url", ""), "content": r.get("content", "")} for r in results],
        ensure_ascii=False,
    )


# --------------------------------------------------------------------------- schemas
class Validation(BaseModel):
    valid: bool = Field(description="True if the query is a valid food/cooking request.")
    reason: str = Field(description="Short explanation when invalid.")


class Ingredients(BaseModel):
    restrictions: List[str] = Field(description="Dietary restrictions declared in the query.")
    ingredients: List[str] = Field(description="Usable ingredients compatible with the restrictions.")


class IngredientItem(BaseModel):
    name: str
    quantity: str


class Recipe(BaseModel):
    title: str
    ingredients: List[IngredientItem]
    equipment: List[str]
    instructions: List[str]
    storage: str


# --------------------------------------------------------------------------- prompts
CHEF = "You are a professional chef who creates personalized recipes."

VALIDATE_PROMPT = CHEF + """
Classify the user's query. It is INVALID if it is empty, offensive, or unrelated to food/cooking/recipes.
Otherwise it is VALID (e.g. any request for a recipe, meal, dish, or cooking with ingredients)."""

IDENTIFY_PROMPT = CHEF + """
Identify the ingredients usable in a recipe. Usable ingredients are:
- every ingredient the user explicitly says they have/want to use in the query, AND
- ALL ingredients from the pantry list.
Rules:
- Exclude ingredients mentioned under negation, absence or hypothesis (e.g. "I don't have eggs",
  "without milk", "I ran out of butter", "if I had cheese", "maybe I'll buy ham").
- Determine dietary restrictions declared in the query (vegan, vegetarian, lactose-free, gluten-free,
  allergies, kosher, halal, etc.). A restriction implies all derived ingredients even if not named
  (e.g. lactose-free excludes milk, butter, cheese, cream, yogurt; vegan excludes meat, fish, eggs,
  dairy, honey, gelatin; gluten-free excludes wheat flour, bread, pasta, barley, rye; vegetarian
  excludes meat, poultry, fish, seafood, gelatin). Remove every incompatible ingredient.
- Treat regional variants and inflected forms as the same ingredient; do not duplicate.
- Do not invent ingredients. If none remains, return an empty list."""

RECIPE_PROMPT = CHEF + """
Create ONE complete, personalized recipe respecting the user's preferences.
- Use the compatible ingredients provided (at least one of them must be in the recipe).
- You may add only basic staples (water, salt, pepper, oil) if compatible with the restrictions.
- NEVER include any ingredient that violates the declared dietary restrictions.
- Reference recipes are only preparation inspiration, not a source of ingredients.
- Give quantities for every ingredient, the equipment, step-by-step instructions and storage advice.
- Answer in the same language as the user's query."""


# --------------------------------------------------------------------------- graph
class State(TypedDict, total=False):
    query: str
    pantry: List[str]
    restrictions: List[str]
    ingredients: List[str]
    references: list
    output: str
    error: Optional[str]


def _llm():
    return ChatOpenAI(model="gpt-4o", temperature=0, timeout=60)


def validate_node(state: State) -> State:
    query = (state.get("query") or "").strip()
    if not query:
        return {"error": "Error: invalid query - the query is empty."}
    res = _llm().with_structured_output(Validation).invoke(
        [SystemMessage(content=VALIDATE_PROMPT), HumanMessage(content=query)]
    )
    if not res.valid:
        return {"error": f"Error: invalid query - {res.reason or 'not related to food or cooking.'}"}
    return {}


def pantry_node(state: State) -> State:
    out = retrieve_pantry.invoke({})
    if str(out).strip().lower().startswith(("error", "erro")):
        return {"error": out if out.startswith("Error:") else "Error: " + out}
    try:
        pantry = json.loads(out)
    except Exception:  # noqa: BLE001
        return {"error": "Error: invalid pantry data returned."}
    return {"pantry": pantry}


def identify_node(state: State) -> State:
    res = _llm().with_structured_output(Ingredients).invoke([
        SystemMessage(content=IDENTIFY_PROMPT),
        HumanMessage(content=f"User query: {state['query']}\n\nPantry ingredients: "
                             f"{json.dumps(state.get('pantry', []), ensure_ascii=False)}"),
    ])
    ingredients = [i.strip() for i in res.ingredients if i and i.strip()]
    if not ingredients:
        return {"restrictions": res.restrictions, "ingredients": [],
                "error": "Error: no ingredient compatible with your dietary restrictions was found "
                         "in the query or in the pantry."}
    return {"restrictions": res.restrictions, "ingredients": ingredients}


def search_node(state: State) -> State:
    out = search_recipes.invoke({"ingredients": state["ingredients"]})
    if str(out).strip().lower().startswith(("error", "erro")):
        return {"error": out if out.startswith("Error:") else "Error: " + out}
    try:
        refs = json.loads(out)
    except Exception:  # noqa: BLE001
        refs = []
    return {"references": refs}


def _format(r: Recipe) -> str:
    lines = [f"Title: {r.title}", "", "Ingredients:"]
    lines += [f"- {i.quantity} {i.name}".replace("  ", " ") for i in r.ingredients]
    lines += ["", "Equipment:"] + [f"- {e}" for e in r.equipment]
    lines += ["", "Instructions:"] + [f"{n}. {s}" for n, s in enumerate(r.instructions, 1)]
    lines += ["", f"Storage: {r.storage}"]
    return "\n".join(lines)


def recipe_node(state: State) -> State:
    refs = "\n".join(
        f"- {r.get('title')} ({r.get('url')}): {str(r.get('content', ''))[:500]}"
        for r in state.get("references", [])
    ) or "None."
    recipe = _llm().with_structured_output(Recipe).invoke([
        SystemMessage(content=RECIPE_PROMPT),
        HumanMessage(content=(
            f"User query: {state['query']}\n"
            f"Declared dietary restrictions: {', '.join(state.get('restrictions', [])) or 'none'}\n"
            f"Compatible ingredients: {', '.join(state['ingredients'])}\n"
            f"Reference recipes:\n{refs}"
        )),
    ])
    if not (recipe.title and recipe.ingredients and recipe.equipment
            and recipe.instructions and recipe.storage):
        return {"error": "Error: the generated recipe is incomplete."}
    return {"output": _format(recipe)}


def _route(next_node: str):
    return lambda s: END if s.get("error") else next_node


def build_graph():
    g = StateGraph(State)
    g.add_node("validate", validate_node)
    g.add_node("pantry", pantry_node)
    g.add_node("identify", identify_node)
    g.add_node("search", search_node)
    g.add_node("recipe", recipe_node)
    g.set_entry_point("validate")
    g.add_conditional_edges("validate", _route("pantry"), ["pantry", END])
    g.add_conditional_edges("pantry", _route("identify"), ["identify", END])
    g.add_conditional_edges("identify", _route("search"), ["search", END])
    g.add_conditional_edges("search", _route("recipe"), ["recipe", END])
    g.add_edge("recipe", END)
    return g.compile()


GRAPH = build_graph()


def agent(query: str) -> str:
    try:
        state = GRAPH.invoke({"query": query or ""})
        if state.get("error"):
            return state["error"]
        return state.get("output") or "Error: no recipe was generated."
    except Exception as exc:  # noqa: BLE001
        return f"Error: execution failure - {exc}"


if __name__ == "__main__":
    import sys
    print(agent(" ".join(sys.argv[1:]) or input("Query: ")))

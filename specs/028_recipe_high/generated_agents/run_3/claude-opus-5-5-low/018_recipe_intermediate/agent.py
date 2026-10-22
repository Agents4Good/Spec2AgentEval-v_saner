"""Personalized recipe generator agent (chef persona) built with LangGraph.

Entrypoint: agent(query: str) -> str
"""
from __future__ import annotations

import json
import os
from typing import List, Optional, TypedDict

import requests
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langgraph.graph import END, StateGraph
from pydantic import BaseModel, Field

MODEL_NAME = "gpt-4o"
HTTP_TIMEOUT = 20
NOTION_VERSION = "2022-06-28"

# --------------------------------------------------------------------------- #
# Tools
# --------------------------------------------------------------------------- #

_TEXT_BLOCKS = (
    "paragraph", "bulleted_list_item", "numbered_list_item", "to_do",
    "heading_1", "heading_2", "heading_3", "toggle", "quote", "callout",
)


def _block_text(block: dict) -> str:
    data = block.get(block.get("type", ""), {}) or {}
    return "".join(rt.get("plain_text", "") for rt in data.get("rich_text", []) or []).strip()


@tool
def retrieve_pantry() -> list | str:
    """Retrieve the complete list of ingredients available in the user's pantry from a Notion page."""
    api_key = os.getenv("NOTION_API_KEY")
    page_id = os.getenv("NOTION_PAGE_ID")
    if not api_key or not page_id:
        return "Error: missing Notion credentials (NOTION_API_KEY / NOTION_PAGE_ID)."
    headers = {"Authorization": f"Bearer {api_key}", "Notion-Version": NOTION_VERSION}
    url = f"https://api.notion.com/v1/blocks/{page_id}/children"
    items: List[str] = []
    cursor: Optional[str] = None
    try:
        while True:
            params = {"page_size": 100}
            if cursor:
                params["start_cursor"] = cursor
            resp = requests.get(url, headers=headers, params=params, timeout=HTTP_TIMEOUT)
            if resp.status_code != 200:
                return f"Error: Notion page inaccessible (HTTP {resp.status_code}): {resp.text[:200]}"
            data = resp.json()
            for block in data.get("results", []):
                if block.get("type") in _TEXT_BLOCKS and not block.get("type", "").startswith("heading"):
                    text = _block_text(block)
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
        return f"Error: failed to access the Notion page: {exc}"
    return items


@tool
def search_recipes(ingredients: List[str]) -> list | str:
    """Search the web (Tavily) for up to 5 reference recipes using the given ingredients.

    Returns a list of {title, url, content}. Results are only a preparation reference.
    """
    api_key = os.getenv("TAVILY_API_KEY")
    if not api_key:
        return "erro: missing Tavily credentials (TAVILY_API_KEY)."
    query = "recipe with " + ", ".join(ingredients)
    try:
        resp = requests.post(
            "https://api.tavily.com/search",
            json={"api_key": api_key, "query": query, "max_results": 5, "search_depth": "basic"},
            headers={"Authorization": f"Bearer {api_key}"},
            timeout=HTTP_TIMEOUT,
        )
        if resp.status_code != 200:
            return f"erro: Tavily inaccessible (HTTP {resp.status_code}): {resp.text[:200]}"
        results = resp.json().get("results", []) or []
    except requests.Timeout:
        return "erro: timeout while searching recipes on Tavily."
    except Exception as exc:  # noqa: BLE001
        return f"erro: failed to search recipes: {exc}"
    return [
        {"title": r.get("title", ""), "url": r.get("url", ""), "content": (r.get("content") or "")[:500]}
        for r in results[:5]
    ]


# --------------------------------------------------------------------------- #
# Structured LLM outputs
# --------------------------------------------------------------------------- #

class Validation(BaseModel):
    is_valid: bool = Field(description="True if the query is a non-empty, non-offensive request related to food/cooking.")
    reason: str = Field(description="Short explanation of the classification.")


class IngredientExtraction(BaseModel):
    dietary_restrictions: List[str] = Field(default_factory=list, description="Dietary restrictions/allergies declared in the query.")
    ingredients: List[str] = Field(default_factory=list, description="Available ingredients compatible with the restrictions.")


class PantryFilter(BaseModel):
    compatible_ingredients: List[str] = Field(default_factory=list)


class Recipe(BaseModel):
    title: str
    ingredients: List[str] = Field(description="Each item with quantity, e.g. '200 g rice'.")
    equipment: List[str]
    instructions: List[str]
    storage: str


PERSONA = (
    "You are an experienced chef who creates personalized recipes adapted to the "
    "ingredients available and to the user's dietary preferences and restrictions."
)

VALIDATION_PROMPT = PERSONA + """
Classify the user's query. It is INVALID if it is empty/blank/meaningless, offensive
(hate, harassment, violence, obscene), or unrelated to food, cooking or recipes.
Otherwise it is VALID (e.g. asking for a recipe, a dish idea, what to cook, with or
without ingredients)."""

EXTRACTION_PROMPT = PERSONA + """
From the user's query:
1. List the dietary restrictions declared (e.g. vegan, vegetarian, lactose-free,
   gluten-free, nut allergy, kosher, halal...).
2. List the ingredients the user says they HAVE / want to use, normalized to their
   singular base name in the query's language (handle regional variants and inflected forms).
Rules:
- EXCLUDE ingredients mentioned under negation, absence or hypothesis
  ("I don't have eggs", "no milk", "out of butter", "if I had cheese", "maybe chicken").
- EXCLUDE ingredients the user does not want or is allergic to.
- A dietary restriction implies ALL its derived ingredients even if not named:
  vegan excludes meat, fish, seafood, eggs, dairy (milk, cheese, butter, cream, yogurt), honey, gelatin;
  vegetarian excludes meat, poultry, fish, seafood, gelatin;
  lactose-free / dairy-free excludes milk, cheese, butter, cream, yogurt, whey;
  gluten-free / celiac excludes wheat, flour, bread, pasta, barley, rye, couscous, regular soy sauce;
  nut allergy excludes all nuts and nut products, etc.
  Remove any ingredient incompatible with the declared restrictions.
- Generic dish names without explicit ingredients produce an empty ingredient list.
Return only compatible ingredients; return an empty list if none."""

PANTRY_FILTER_PROMPT = PERSONA + """
Given the pantry ingredients and the dietary restrictions declared in the user's query,
return ONLY the pantry ingredients compatible with those restrictions (a restriction
implies all its derived ingredients). Keep names as given. If no restriction, keep all."""

RECIPE_PROMPT = PERSONA + """
Create ONE complete, original recipe for the user.
Rules:
- Use the provided compatible ingredients as the base; the recipe MUST contain at least
  one of them. You may add common staples (water, salt, oil, spices) only if they respect
  every dietary restriction.
- NO ingredient may violate the declared dietary restrictions (including derived ingredients).
- Respect the user's preferences (cuisine, time, style, servings).
- Reference recipes (if any) are only preparation inspiration, not a source of ingredients.
- Every field must be non-empty: title, ingredients (with quantities), equipment,
  step-by-step instructions, storage recommendations."""


def _llm() -> ChatOpenAI:
    return ChatOpenAI(model=MODEL_NAME, temperature=0, timeout=60, max_retries=2)


# --------------------------------------------------------------------------- #
# Graph
# --------------------------------------------------------------------------- #

class State(TypedDict, total=False):
    query: str
    is_valid: bool
    restrictions: List[str]
    ingredients: List[str]
    pantry: List[str]
    references: list
    response: str


def validate_node(state: State) -> State:
    query = (state.get("query") or "").strip()
    if not query:
        return {"is_valid": False, "response": "Error: invalid query - the query is empty."}
    try:
        result = _llm().with_structured_output(Validation).invoke(
            [SystemMessage(content=VALIDATION_PROMPT), HumanMessage(content=query)]
        )
    except Exception as exc:  # noqa: BLE001
        return {"is_valid": False, "response": f"Error: model failure during validation: {exc}"}
    if not result.is_valid:
        return {"is_valid": False, "response": f"Error: invalid query - {result.reason}"}
    return {"is_valid": True}


def identify_node(state: State) -> State:
    try:
        result = _llm().with_structured_output(IngredientExtraction).invoke(
            [SystemMessage(content=EXTRACTION_PROMPT), HumanMessage(content=state["query"])]
        )
    except Exception as exc:  # noqa: BLE001
        return {"response": f"Error: model failure during ingredient identification: {exc}"}
    seen, ingredients = set(), []
    for ing in result.ingredients:
        key = ing.strip().lower()
        if key and key not in seen:
            seen.add(key)
            ingredients.append(ing.strip())
    return {"ingredients": ingredients, "restrictions": result.dietary_restrictions}


def search_node(state: State) -> State:
    try:
        result = search_recipes.invoke({"ingredients": state["ingredients"]})
    except Exception as exc:  # noqa: BLE001
        return {"response": f"Error: recipe search failed: {exc}"}
    if isinstance(result, str):
        msg = result.split(":", 1)[1].strip() if result.lower().startswith("erro") and ":" in result else result
        return {"response": f"Error: recipe search failed - {msg}"}
    return {"references": result}


def pantry_node(state: State) -> State:
    try:
        result = retrieve_pantry.invoke({})
    except Exception as exc:  # noqa: BLE001
        return {"response": f"Error: pantry lookup failed: {exc}"}
    if isinstance(result, str):
        msg = result.split(":", 1)[1].strip() if ":" in result else result
        return {"response": f"Error: pantry lookup failed - {msg}"}
    if not result:
        return {"pantry": [], "response": "Error: lack of ingredients - the pantry is empty."}
    try:
        filtered = _llm().with_structured_output(PantryFilter).invoke(
            [
                SystemMessage(content=PANTRY_FILTER_PROMPT),
                HumanMessage(content=(
                    f"User query: {state['query']}\n"
                    f"Declared restrictions: {state.get('restrictions') or 'none'}\n"
                    f"Pantry: {json.dumps(result, ensure_ascii=False)}"
                )),
            ]
        )
    except Exception as exc:  # noqa: BLE001
        return {"response": f"Error: model failure while filtering pantry: {exc}"}
    pantry_lower = {p.lower() for p in result}
    compatible = [i for i in filtered.compatible_ingredients if i.lower() in pantry_lower]
    if not compatible:
        return {"pantry": [], "response": "Error: lack of ingredients - no pantry ingredient is compatible with your dietary restrictions."}
    return {"pantry": compatible}


def format_recipe(recipe: Recipe) -> str:
    lines = [f"Title: {recipe.title}", "", "Ingredients:"]
    lines += [f"- {i}" for i in recipe.ingredients]
    lines += ["", "Equipment:"] + [f"- {e}" for e in recipe.equipment]
    lines += ["", "Instructions:"] + [f"{n}. {s}" for n, s in enumerate(recipe.instructions, 1)]
    lines += ["", f"Storage: {recipe.storage}"]
    return "\n".join(lines)


def generate_node(state: State) -> State:
    base = state.get("ingredients") or state.get("pantry") or []
    refs = state.get("references") or []
    ref_txt = "\n".join(f"- {r['title']} ({r['url']}): {r['content']}" for r in refs) or "none"
    prompt = (
        f"User query: {state['query']}\n"
        f"Declared dietary restrictions: {state.get('restrictions') or 'none'}\n"
        f"Compatible ingredients (use at least one): {', '.join(base)}\n"
        f"Reference recipes:\n{ref_txt}"
    )
    try:
        recipe = _llm().with_structured_output(Recipe).invoke(
            [SystemMessage(content=RECIPE_PROMPT), HumanMessage(content=prompt)]
        )
    except Exception as exc:  # noqa: BLE001
        return {"response": f"Error: model failure during recipe generation: {exc}"}
    if not all([recipe.title.strip(), recipe.ingredients, recipe.equipment, recipe.instructions, recipe.storage.strip()]):
        return {"response": "Error: model failure - generated recipe is incomplete."}
    return {"response": format_recipe(recipe)}


def _after_validate(state: State) -> str:
    return "identify" if state.get("is_valid") else END


def _after_identify(state: State) -> str:
    if state.get("response"):
        return END
    return "search" if state.get("ingredients") else "pantry"


def _after_tool(state: State) -> str:
    return END if state.get("response") else "generate"


def build_graph():
    g = StateGraph(State)
    g.add_node("validate", validate_node)
    g.add_node("identify", identify_node)
    g.add_node("search", search_node)
    g.add_node("pantry", pantry_node)
    g.add_node("generate", generate_node)
    g.set_entry_point("validate")
    g.add_conditional_edges("validate", _after_validate, {"identify": "identify", END: END})
    g.add_conditional_edges("identify", _after_identify, {"search": "search", "pantry": "pantry", END: END})
    g.add_conditional_edges("search", _after_tool, {"generate": "generate", END: END})
    g.add_conditional_edges("pantry", _after_tool, {"generate": "generate", END: END})
    g.add_edge("generate", END)
    return g.compile()


def agent(query: str) -> str:
    try:
        final = build_graph().invoke({"query": query or ""})
        return final.get("response") or "Error: no response produced."
    except Exception as exc:  # noqa: BLE001
        return f"Error: unexpected execution failure: {exc}"


if __name__ == "__main__":
    import sys
    print(agent(" ".join(sys.argv[1:]) or "I have rice and tomatoes, I'm vegan"))

"""Personalized recipe generator agent (chef persona) built with LangGraph."""
import json
import os
from typing import List, Optional, TypedDict

import requests
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langgraph.graph import END, StateGraph
from pydantic import BaseModel, Field

load_dotenv()

TIMEOUT = 20
NOTION_VERSION = "2022-06-28"
TEXT_BLOCKS = (
    "paragraph", "bulleted_list_item", "numbered_list_item", "to_do",
    "toggle", "heading_1", "heading_2", "heading_3", "quote", "callout",
)


# ----------------------------------------------------------------------------
# Tools
# ----------------------------------------------------------------------------
def _split_items(text: str) -> List[str]:
    items = []
    for line in text.replace(";", "\n").splitlines():
        for part in line.split(","):
            part = part.strip().strip("-•*").strip()
            if part:
                items.append(part)
    return items


@tool
def retrieve_pantry() -> list | str:
    """Retrieve the complete list of ingredients available in the user's pantry (Notion page).
    Returns a list of ingredient names (empty list for an empty pantry) or a string starting
    with 'Error:' on failure."""
    api_key = os.getenv("NOTION_API_KEY")
    page_id = os.getenv("NOTION_PAGE_ID")
    if not api_key or not page_id:
        return "Error: missing Notion credentials (NOTION_API_KEY / NOTION_PAGE_ID)."
    headers = {"Authorization": f"Bearer {api_key}", "Notion-Version": NOTION_VERSION}
    url = f"https://api.notion.com/v1/blocks/{page_id}/children"
    ingredients: List[str] = []
    cursor = None
    try:
        while True:
            params = {"page_size": 100}
            if cursor:
                params["start_cursor"] = cursor
            resp = requests.get(url, headers=headers, params=params, timeout=TIMEOUT)
            if resp.status_code != 200:
                return f"Error: could not access the Notion page (HTTP {resp.status_code}): {resp.text[:200]}"
            data = resp.json()
            for block in data.get("results", []):
                btype = block.get("type")
                if btype not in TEXT_BLOCKS or btype.startswith("heading"):
                    continue
                rich = block.get(btype, {}).get("rich_text", [])
                text = "".join(r.get("plain_text", "") for r in rich)
                ingredients.extend(_split_items(text))
            if not data.get("has_more"):
                break
            cursor = data.get("next_cursor")
    except requests.Timeout:
        return "Error: timeout while accessing the Notion page."
    except Exception as exc:  # noqa: BLE001
        return f"Error: failed to retrieve pantry: {exc}"
    return ingredients


@tool
def search_recipes(ingredients: List[str]) -> list | str:
    """Search the web (Tavily) for up to 5 reference recipes using the given ingredients.
    Returns a list of {title, url, content} (possibly empty) or a string starting with 'erro:'."""
    api_key = os.getenv("TAVILY_API_KEY")
    if not api_key:
        return "erro: missing Tavily credentials (TAVILY_API_KEY)."
    query = "recipe with " + ", ".join(ingredients)
    try:
        resp = requests.post(
            "https://api.tavily.com/search",
            json={"api_key": api_key, "query": query, "max_results": 5, "search_depth": "basic"},
            headers={"Authorization": f"Bearer {api_key}"},
            timeout=TIMEOUT,
        )
        if resp.status_code != 200:
            return f"erro: Tavily request failed (HTTP {resp.status_code}): {resp.text[:200]}"
        results = resp.json().get("results", []) or []
    except requests.Timeout:
        return "erro: timeout while searching recipes."
    except Exception as exc:  # noqa: BLE001
        return f"erro: failed to search recipes: {exc}"
    return [
        {"title": r.get("title", ""), "url": r.get("url", ""), "content": (r.get("content") or "")[:500]}
        for r in results[:5]
    ]


def _is_tool_error(result) -> bool:
    return isinstance(result, str) and result.strip().lower().startswith(("erro:", "error:"))


# ----------------------------------------------------------------------------
# Structured LLM outputs
# ----------------------------------------------------------------------------
class Analysis(BaseModel):
    is_valid: bool = Field(description="False if the query is empty, offensive, or unrelated to food/cooking.")
    reason: str = Field(description="Short explanation of the classification.")
    dietary_restrictions: List[str] = Field(default_factory=list)
    ingredients: List[str] = Field(
        default_factory=list,
        description="Available ingredients mentioned in the query, compatible with the restrictions.",
    )


class PantryFilter(BaseModel):
    compatible_ingredients: List[str] = Field(default_factory=list)


class IngredientLine(BaseModel):
    name: str
    quantity: str


class Recipe(BaseModel):
    title: str
    ingredients: List[IngredientLine]
    equipment: List[str]
    instructions: List[str]
    storage: str


CHEF = (
    "You are an experienced chef who creates personalized recipes. "
    "Invariants you must always respect:\n"
    "- A dietary restriction implies ALL of its derived ingredients, even if not mentioned "
    "(e.g. vegan excludes meat, fish, eggs, dairy, honey, butter, gelatin; lactose-free excludes "
    "milk, cheese, butter, cream, yogurt; gluten-free excludes wheat, flour, bread, pasta, barley, rye; "
    "vegetarian excludes meat, fish, seafood, gelatin; allergies exclude the allergen and its derivatives).\n"
    "- An ingredient mentioned under negation, absence, or hypothesis (\"I don't have eggs\", "
    "\"without milk\", \"if I had chicken\", \"I ran out of rice\") is NOT an available ingredient."
)

ANALYSIS_PROMPT = CHEF + """

Analyze the user's query.
1. Validity: invalid if empty/blank, offensive, or unrelated to food/cooking. Otherwise valid.
2. List the dietary restrictions declared (empty if none).
3. List the ingredients the user actually HAS/wants to use, normalized to their singular base name
   in the language of the query (handle regional variants and inflected forms). Exclude ingredients
   under negation, absence or hypothesis, and exclude ingredients incompatible with the declared
   restrictions. Do NOT include generic categories or dish names unless they are ingredients.
   Empty list if none."""


def _llm():
    return ChatOpenAI(model="gpt-4o", temperature=0, timeout=60, max_retries=2)


# ----------------------------------------------------------------------------
# Graph
# ----------------------------------------------------------------------------
class State(TypedDict, total=False):
    query: str
    valid: bool
    restrictions: List[str]
    ingredients: List[str]
    pantry: List[str]
    references: list
    response: str


def validate_and_identify(state: State) -> State:  # B1 + B2
    query = (state.get("query") or "").strip()
    if not query:
        return {"valid": False, "response": "Error: invalid query - the query is empty."}
    result: Analysis = _llm().with_structured_output(Analysis).invoke(
        [SystemMessage(content=ANALYSIS_PROMPT), HumanMessage(content=query)]
    )
    if not result.is_valid:
        return {"valid": False, "response": f"Error: invalid query - {result.reason}"}
    seen, ingredients = set(), []
    for ing in result.ingredients:
        key = ing.strip().lower()
        if key and key not in seen:
            seen.add(key)
            ingredients.append(ing.strip())
    return {"valid": True, "restrictions": result.dietary_restrictions, "ingredients": ingredients}


def search_node(state: State) -> State:  # B3
    result = search_recipes.invoke({"ingredients": state["ingredients"]})
    if _is_tool_error(result):
        return {"response": f"Error: recipe search failed - {result.split(':', 1)[1].strip()}"}
    return {"references": result if isinstance(result, list) else []}


def pantry_node(state: State) -> State:  # B4 + B5
    result = retrieve_pantry.invoke({})
    if _is_tool_error(result):
        return {"response": f"Error: pantry lookup failed - {result.split(':', 1)[1].strip()}"}
    items = result if isinstance(result, list) else []
    compatible: List[str] = []
    if items:
        restrictions = state.get("restrictions") or []
        if restrictions:
            out: PantryFilter = _llm().with_structured_output(PantryFilter).invoke([
                SystemMessage(content=CHEF + "\n\nFrom the pantry list, return only the items compatible "
                              "with ALL the dietary restrictions. Keep names exactly as given."),
                HumanMessage(content=json.dumps({"restrictions": restrictions, "pantry": items}, ensure_ascii=False)),
            ])
            allowed = {i.lower() for i in items}
            compatible = [i for i in out.compatible_ingredients if i.lower() in allowed]
        else:
            compatible = items
    if not compatible:
        return {"response": "Error: no pantry ingredients compatible with your dietary restrictions are available."}
    return {"pantry": compatible}


def generate_node(state: State) -> State:  # B7
    base = state.get("ingredients") or state.get("pantry") or []
    payload = {
        "user_query": state["query"],
        "dietary_restrictions": state.get("restrictions") or [],
        "available_ingredients": base,
        "source": "query" if state.get("ingredients") else "pantry",
        "reference_recipes": state.get("references") or [],
    }
    recipe: Recipe = _llm().with_structured_output(Recipe).invoke([
        SystemMessage(content=CHEF + "\n\nCreate ONE complete recipe matching the user's preferences. "
                      "It must use at least one of the available_ingredients (use as many as sensible). "
                      "Reference recipes are only preparation guidance, not ingredient sources. You may add "
                      "common basics (salt, water, oil, spices) only if compatible with every restriction. "
                      "Never include an ingredient that violates a restriction or that the user said they do "
                      "not have. Give quantities for every ingredient. Write in the language of the query."),
        HumanMessage(content=json.dumps(payload, ensure_ascii=False)),
    ])
    lower = " ".join(i.name.lower() for i in recipe.ingredients)
    if not any(b.lower() in lower for b in base):
        recipe.ingredients.insert(0, IngredientLine(name=base[0], quantity="to taste"))
    if not (recipe.title and recipe.ingredients and recipe.equipment and recipe.instructions and recipe.storage):
        return {"response": "Error: the generated recipe is incomplete."}
    return {"response": render_recipe(recipe)}


def render_recipe(r: Recipe) -> str:
    lines = [f"Title: {r.title}", "", "Ingredients:"]
    lines += [f"- {i.quantity} {i.name}".replace("  ", " ") for i in r.ingredients]
    lines += ["", "Equipment:"] + [f"- {e}" for e in r.equipment]
    lines += ["", "Instructions:"] + [f"{n}. {s}" for n, s in enumerate(r.instructions, 1)]
    lines += ["", "Storage:", r.storage]
    return "\n".join(lines)


def _route_after_analysis(state: State) -> str:
    if not state.get("valid"):
        return END
    return "search" if state.get("ingredients") else "pantry"


def _route_to_generate(state: State) -> str:
    return END if state.get("response") else "generate"


def build_graph():
    g = StateGraph(State)
    g.add_node("analyze", validate_and_identify)
    g.add_node("search", search_node)
    g.add_node("pantry", pantry_node)
    g.add_node("generate", generate_node)
    g.set_entry_point("analyze")
    g.add_conditional_edges("analyze", _route_after_analysis, {"search": "search", "pantry": "pantry", END: END})
    g.add_conditional_edges("search", _route_to_generate, {"generate": "generate", END: END})
    g.add_conditional_edges("pantry", _route_to_generate, {"generate": "generate", END: END})
    g.add_edge("generate", END)
    return g.compile()


def agent(query: str) -> str:
    """Entrypoint: returns a recipe string or a message starting with 'Error:'."""
    try:
        if not os.getenv("OPENAI_API_KEY"):
            return "Error: missing OpenAI credentials (OPENAI_API_KEY)."
        state = build_graph().invoke({"query": query if isinstance(query, str) else ""})
        response = state.get("response")
        return response if response else "Error: no response was produced."
    except Exception as exc:  # B6
        return f"Error: execution failure - {exc}"


if __name__ == "__main__":
    import sys
    print(agent(" ".join(sys.argv[1:]) or input("Query: ")))

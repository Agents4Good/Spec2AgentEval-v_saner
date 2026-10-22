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

MODEL_NAME = "gpt-4o"
HTTP_TIMEOUT = 20

CHEF_PERSONA = (
    "You are an experienced chef whose role is to create personalized recipes "
    "adapted to the user's available ingredients, preferences and dietary restrictions."
)

RESTRICTION_RULES = (
    "Invariant: a dietary restriction implies ALL of its derived ingredients, even if the user "
    "did not mention them (e.g. vegan excludes meat, fish, seafood, eggs, dairy such as milk, butter, "
    "cheese, cream, yogurt, and honey, gelatin; vegetarian excludes meat, poultry, fish, seafood, gelatin; "
    "lactose-free excludes milk, butter, cheese, cream, yogurt; gluten-free excludes wheat, flour, "
    "bread, pasta, barley, rye, regular soy sauce; nut allergy excludes all nuts and nut-derived products, etc.)."
)


# ----------------------------------------------------------------------------- tools
def _rich_text(block: dict) -> str:
    data = block.get(block.get("type", ""), {}) or {}
    return "".join(t.get("plain_text", "") for t in data.get("rich_text", []) or []).strip()


@tool
def retrieve_pantry() -> list | str:
    """Retrieve the complete list of ingredients available in the user's pantry from a Notion page."""
    api_key = os.getenv("NOTION_API_KEY")
    page_id = os.getenv("NOTION_PAGE_ID")
    if not api_key or not page_id:
        return "Error: missing Notion credentials (NOTION_API_KEY / NOTION_PAGE_ID)."
    headers = {"Authorization": f"Bearer {api_key}", "Notion-Version": "2022-06-28"}
    items: List[str] = []
    cursor = None
    try:
        while True:
            params = {"page_size": 100}
            if cursor:
                params["start_cursor"] = cursor
            resp = requests.get(
                f"https://api.notion.com/v1/blocks/{page_id}/children",
                headers=headers, params=params, timeout=HTTP_TIMEOUT,
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
                for line in text.splitlines():
                    for part in line.split(","):
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
            timeout=HTTP_TIMEOUT,
        )
        if resp.status_code != 200:
            return f"erro: Tavily request failed (HTTP {resp.status_code}): {resp.text[:200]}"
        results = resp.json().get("results", []) or []
    except requests.Timeout:
        return "erro: timeout while searching recipes on Tavily."
    except Exception as exc:  # noqa: BLE001
        return f"erro: failed to search recipes: {exc}"
    return [
        {"title": r.get("title", ""), "url": r.get("url", ""), "content": (r.get("content") or "")[:500]}
        for r in results[:5]
    ]


# ----------------------------------------------------------------------------- schemas
class Validation(BaseModel):
    is_valid: bool = Field(description="True if the query is a non-empty, non-offensive food/cooking request.")
    reason: str = Field(description="Short explanation of the classification.")


class Identification(BaseModel):
    dietary_restrictions: List[str] = Field(default_factory=list)
    ingredients: List[str] = Field(
        default_factory=list,
        description="Available ingredients from the query compatible with the restrictions.",
    )


class PantryFilter(BaseModel):
    compatible_ingredients: List[str] = Field(default_factory=list)


class RecipeIngredient(BaseModel):
    name: str
    quantity: str


class Recipe(BaseModel):
    title: str
    ingredients: List[RecipeIngredient]
    equipment: List[str]
    instructions: List[str]
    storage: str


class State(TypedDict, total=False):
    query: str
    valid: bool
    restrictions: List[str]
    ingredients: List[str]
    pantry: List[str]
    references: list
    response: str


def _llm() -> ChatOpenAI:
    return ChatOpenAI(model=MODEL_NAME, temperature=0, timeout=60, max_retries=2)


# ----------------------------------------------------------------------------- nodes
def validate_node(state: State) -> State:
    query = (state.get("query") or "").strip()
    if not query:
        return {"valid": False, "response": "Error: invalid query: the query is empty."}
    result: Validation = _llm().with_structured_output(Validation).invoke([
        SystemMessage(content=(
            "Classify the user query. It is INVALID if it is empty/meaningless, offensive, or unrelated to "
            "food/cooking/recipes. Otherwise it is VALID (including requests for a recipe without ingredients)."
        )),
        HumanMessage(content=query),
    ])
    if not result.is_valid:
        return {"valid": False, "response": f"Error: invalid query: {result.reason}"}
    return {"valid": True}


def identify_node(state: State) -> State:
    result: Identification = _llm().with_structured_output(Identification).invoke([
        SystemMessage(content=(
            CHEF_PERSONA + "\nExtract the declared dietary restrictions and the ingredients the user HAS "
            "available/wants to use from the query. Rules:\n"
            "- Normalize regional variants and inflected forms to a canonical singular name, in the query language.\n"
            "- Exclude ingredients mentioned under negation, absence or hypothesis (e.g. 'I don't have eggs', "
            "'no milk', 'if I had cheese', 'without onion').\n"
            "- Exclude ingredients incompatible with the declared dietary restrictions.\n"
            "- Do not invent ingredients; dish names or generic preferences are not ingredients unless explicit.\n"
            + RESTRICTION_RULES
        )),
        HumanMessage(content=state["query"]),
    ])
    return {
        "restrictions": result.dietary_restrictions,
        "ingredients": [i.strip() for i in result.ingredients if i.strip()],
    }


def search_node(state: State) -> State:
    result = search_recipes.invoke({"ingredients": state["ingredients"]})
    if isinstance(result, str):
        return {"response": f"Error: recipe search failed: {result.split(':', 1)[-1].strip()}"}
    return {"references": result}


def pantry_node(state: State) -> State:
    result = retrieve_pantry.invoke({})
    if isinstance(result, str):
        msg = result.split(":", 1)[-1].strip() if result.lower().startswith(("error", "erro")) else result
        return {"response": f"Error: pantry lookup failed: {msg}"}
    if not result:
        return {"pantry": [], "response": "Error: no compatible ingredients: the pantry is empty."}
    filtered: PantryFilter = _llm().with_structured_output(PantryFilter).invoke([
        SystemMessage(content=(
            "From the pantry list, return only the items compatible with the dietary restrictions declared "
            "in the user query (keep their names exactly as in the pantry). If no restriction is declared, "
            "return all items. " + RESTRICTION_RULES
        )),
        HumanMessage(content=f"User query: {state['query']}\nPantry: {json.dumps(result, ensure_ascii=False)}"),
    ])
    pantry = [p for p in filtered.compatible_ingredients if p.strip()]
    if not pantry:
        return {"pantry": [], "response": (
            "Error: no compatible ingredients: the pantry has no ingredient compatible with the dietary restrictions."
        )}
    return {"pantry": pantry}


def generate_node(state: State) -> State:
    base = state.get("ingredients") or state.get("pantry") or []
    refs = state.get("references") or []
    recipe: Recipe = _llm().with_structured_output(Recipe).invoke([
        SystemMessage(content=(
            CHEF_PERSONA + "\nCreate one complete recipe answering the user's preferences, in the language of "
            "the query. It MUST use at least one of the base ingredients (use their names) and must NOT contain "
            "any ingredient that violates the declared dietary restrictions (including derived ingredients). "
            "You may add common seasonings/basics compatible with the restrictions. Reference recipes are only "
            "a preparation reference, not a source of base ingredients. All fields must be non-empty; include "
            "quantities for every ingredient. " + RESTRICTION_RULES
        )),
        HumanMessage(content=(
            f"User query: {state['query']}\n"
            f"Declared restrictions: {state.get('restrictions') or 'none'}\n"
            f"Base ingredients: {json.dumps(base, ensure_ascii=False)}\n"
            f"Reference recipes: {json.dumps(refs, ensure_ascii=False)}"
        )),
    ])
    if not (recipe.title and recipe.ingredients and recipe.equipment and recipe.instructions and recipe.storage):
        return {"response": "Error: model failure: generated recipe is incomplete."}
    lines = [f"Title: {recipe.title}", "", "Ingredients:"]
    lines += [f"- {i.quantity} {i.name}".strip() for i in recipe.ingredients]
    lines += ["", "Equipment:"] + [f"- {e}" for e in recipe.equipment]
    lines += ["", "Instructions:"] + [f"{n}. {s}" for n, s in enumerate(recipe.instructions, 1)]
    lines += ["", f"Storage: {recipe.storage}"]
    return {"response": "\n".join(lines)}


# ----------------------------------------------------------------------------- graph
def _after_validate(state: State) -> str:
    return "identify" if state.get("valid") else END


def _after_identify(state: State) -> str:
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
    g.add_conditional_edges("identify", _after_identify, {"search": "search", "pantry": "pantry"})
    g.add_conditional_edges("search", _after_tool, {"generate": "generate", END: END})
    g.add_conditional_edges("pantry", _after_tool, {"generate": "generate", END: END})
    g.add_edge("generate", END)
    return g.compile()


def agent(query: str) -> str:
    """Entrypoint: returns the recipe or a message beginning with 'Error:'."""
    try:
        if not os.getenv("OPENAI_API_KEY"):
            return "Error: missing OpenAI credentials (OPENAI_API_KEY)."
        result = build_graph().invoke({"query": query or ""})
        return result.get("response") or "Error: execution produced no response."
    except Exception as exc:  # noqa: BLE001
        return f"Error: execution failure: {exc}"


if __name__ == "__main__":
    import sys
    print(agent(" ".join(sys.argv[1:]) or input("Query: ")))

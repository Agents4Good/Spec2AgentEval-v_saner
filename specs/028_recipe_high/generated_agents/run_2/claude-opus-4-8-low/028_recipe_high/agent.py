"""Personalized recipe generator agent (high-complexity spec, 028).

Persona: a chef that creates personalized recipes adapted to the ingredients
provided in the user's query and available in their pantry, while respecting
the user's declared dietary preferences and restrictions. The agent returns a
complete recipe with title, ingredients, equipment, instructions and storage
recommendations.

Interface contract
------------------
Model:      GPT-4o
Framework:  LangGraph
GenAI tool: ChatOpenAI (langchain-openai)
Tools:      retrieve_pantry, search_recipes -- both LangChain ``@tool``
            objects, executed exclusively through their ``.invoke`` interface
            (never by calling the implementing function directly).
Entrypoint: agent(query: str) -> str

Behavioral graph
----------------
B1 validate query -> B2 retrieve_pantry -> B3 identify compatible ingredients
    -> (non-empty) B4 search_recipes -> B7 generate recipe
    -> (empty)     B5 Error: lack of ingredients
Any tool/model failure is funnelled to B6 (Error: execution failure).
"""

from __future__ import annotations

import json
import os
from typing import List, Optional, TypedDict

import requests
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langgraph.graph import END, StateGraph


# --------------------------------------------------------------------------- #
# Tools (LangChain @tool, invoked exclusively through .invoke)                 #
# --------------------------------------------------------------------------- #
@tool
def retrieve_pantry() -> List[str]:
    """Obtain the ingredients available in the user's pantry from a Notion page.

    Takes no input. Returns the complete list of pantry ingredients; an empty
    pantry returns an empty list. On missing credentials, an inaccessible page,
    or a timeout, returns a string beginning with ``Error:`` followed by a
    message explaining the failure.
    """
    api_key = os.environ.get("NOTION_API_KEY")
    page_id = os.environ.get("NOTION_PAGE_ID")

    if not api_key:
        return "Error: missing NOTION_API_KEY credential to access the pantry."
    if not page_id:
        return "Error: missing NOTION_PAGE_ID to locate the pantry page."

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Notion-Version": "2022-06-28",
    }
    base_url = f"https://api.notion.com/v1/blocks/{page_id}/children?page_size=100"

    try:
        ingredients: List[str] = []
        next_cursor: Optional[str] = None
        while True:
            url = base_url
            if next_cursor:
                url = f"{base_url}&start_cursor={next_cursor}"
            resp = requests.get(url, headers=headers, timeout=30)
            if resp.status_code != 200:
                return (
                    "Error: could not access the Notion pantry page "
                    f"(status {resp.status_code})."
                )
            data = resp.json()
            for block in data.get("results", []):
                block_type = block.get("type")
                rich_text = (block.get(block_type, {}) or {}).get("rich_text", [])
                text = "".join(rt.get("plain_text", "") for rt in rich_text).strip()
                if text:
                    ingredients.append(text)
            if data.get("has_more"):
                next_cursor = data.get("next_cursor")
            else:
                break
        return ingredients
    except requests.exceptions.Timeout:
        return "Error: timeout while accessing the Notion pantry page."
    except Exception as exc:  # pragma: no cover - defensive
        return f"Error: failed to retrieve the pantry ({exc})."


@tool
def search_recipes(ingredients: List[str]) -> List[dict]:
    """Search the web (via Tavily) for recipes to use as a preparation reference.

    Input is the list of usable ingredients (compatible with the dietary
    restrictions) derived from the pantry and the query. The recipes found are
    a preparation *reference* only -- neither a source of base ingredients nor
    the final recipe. Returns up to 5 results, each a dict with ``title``,
    ``url`` and ``content`` (excerpt); no results returns an empty list. On
    missing credentials, an inaccessible page, or a timeout, returns a string
    beginning with ``Error:`` followed by a message explaining the failure.
    """
    api_key = os.environ.get("TAVILY_API_KEY")
    if not api_key:
        return "Error: missing TAVILY_API_KEY credential to search for recipes."

    query = "recipe using " + ", ".join(ingredients)
    try:
        resp = requests.post(
            "https://api.tavily.com/search",
            json={
                "api_key": api_key,
                "query": query,
                "max_results": 5,
                "search_depth": "basic",
            },
            timeout=30,
        )
        if resp.status_code != 200:
            return (
                "Error: could not reach the recipe search service "
                f"(status {resp.status_code})."
            )
        data = resp.json()
        results: List[dict] = []
        for item in data.get("results", [])[:5]:
            results.append(
                {
                    "title": item.get("title", ""),
                    "url": item.get("url", ""),
                    "content": item.get("content", ""),
                }
            )
        return results
    except requests.exceptions.Timeout:
        return "Error: timeout while searching for recipes."
    except Exception as exc:  # pragma: no cover - defensive
        return f"Error: failed to search for recipes ({exc})."


# --------------------------------------------------------------------------- #
# LLM (GPT-4o via ChatOpenAI)                                                  #
# --------------------------------------------------------------------------- #
def _get_llm() -> ChatOpenAI:
    return ChatOpenAI(model="gpt-4o", temperature=0)


def _is_error(value) -> bool:
    """A tool returned a failure message (begins with 'Error:' or 'erro:')."""
    return isinstance(value, str) and (
        value.lower().startswith("error:") or value.lower().startswith("erro:")
    )


def _normalize_error(value: str) -> str:
    """Convert a tool failure message into a canonical ``Error:`` message."""
    _, _, rest = value.partition(":")
    return "Error: " + rest.strip()


# --------------------------------------------------------------------------- #
# Graph state                                                                  #
# --------------------------------------------------------------------------- #
class RecipeState(TypedDict, total=False):
    query: str
    valid: bool
    pantry_ingredients: List[str]
    ingredients: List[str]
    search_results: list
    response: str
    error: str


# --------------------------------------------------------------------------- #
# Behavior nodes                                                               #
# --------------------------------------------------------------------------- #
def b1_validate(state: RecipeState) -> RecipeState:
    """B1 - classify the query as valid or invalid; stop if invalid."""
    query = (state.get("query") or "").strip()
    if not query:
        return {"valid": False, "error": "Error: the query is empty."}

    llm = _get_llm()
    prompt = (
        "You are a validator for a recipe-generation assistant. "
        "Classify the user's query as VALID or INVALID.\n"
        "A query is INVALID if it is empty, offensive, or unrelated to "
        "food/cooking. Any query that meets none of these criteria is VALID.\n"
        'Respond ONLY with a JSON object: {"valid": true|false, '
        '"reason": "short explanation"}.\n\n'
        f"Query: {query}"
    )
    try:
        raw = llm.invoke(prompt).content
        data = _parse_json(raw)
        if bool(data.get("valid")):
            return {"valid": True}
        reason = data.get("reason") or "the query is not related to food/cooking."
        return {"valid": False, "error": f"Error: invalid query - {reason}"}
    except Exception as exc:
        return {"valid": False, "error": f"Error: failed to validate the query ({exc})."}


def b2_pantry(state: RecipeState) -> RecipeState:
    """B2 - retrieve the ingredients available in the user's pantry.

    ``retrieve_pantry`` is the first tool called and is called exactly once.
    """
    try:
        pantry = retrieve_pantry.invoke({})
    except Exception as exc:  # B6
        return {"error": f"Error: failed to retrieve the pantry ({exc})."}
    if _is_error(pantry):  # B6 - tool returned a failure message
        return {"error": _normalize_error(str(pantry))}
    return {"pantry_ingredients": list(pantry) if pantry else []}


def b3_identify(state: RecipeState) -> RecipeState:
    """B3 - identify ingredients usable in the recipe.

    Usable ingredients are those mentioned in the query plus all those returned
    by the pantry, keeping only the ones compatible with the declared dietary
    restrictions.
    """
    query = state["query"]
    pantry = state.get("pantry_ingredients", [])
    llm = _get_llm()
    prompt = (
        "Identify the food ingredients usable in a recipe for the user.\n"
        "Usable ingredients are:\n"
        "  (a) ingredients explicitly mentioned in the query as AVAILABLE, and\n"
        "  (b) ALL ingredients returned from the pantry.\n"
        "Rules:\n"
        "- Consider regional variants and inflected/plural forms.\n"
        "- EXCLUDE any ingredient mentioned under negation, absence, or "
        "hypothesis (e.g. 'without X', 'I ran out of X', 'if only I had X'); "
        "these are not available ingredients.\n"
        "- A declared dietary restriction implies ALL of its derived "
        "ingredients, even those not mentioned by the user. EXCLUDE every "
        "ingredient incompatible with the declared dietary restrictions "
        "(this applies to BOTH query and pantry ingredients).\n"
        'Respond ONLY with a JSON object: {"ingredients": ["..."]}. '
        "If no ingredient is identified or compatible, return an empty list.\n\n"
        f"Query: {query}\n"
        f"Pantry ingredients: {json.dumps(pantry)}"
    )
    try:
        raw = llm.invoke(prompt).content
        data = _parse_json(raw)
        ingredients = [
            str(i).strip() for i in (data.get("ingredients") or []) if str(i).strip()
        ]
        return {"ingredients": ingredients}
    except Exception as exc:  # B6
        return {"error": f"Error: failed to identify ingredients ({exc})."}


def b4_search(state: RecipeState) -> RecipeState:
    """B4 - search reference recipes for the ingredients identified in B3.

    ``search_recipes`` is called exactly once, only after ``retrieve_pantry``
    has completed, with the list of ingredients from B3.
    """
    try:
        results = search_recipes.invoke({"ingredients": state["ingredients"]})
    except Exception as exc:  # B6
        return {"error": f"Error: failed to search for recipes ({exc})."}
    if _is_error(results):  # B6 - tool returned a failure message
        return {"error": _normalize_error(str(results))}
    return {"search_results": results}


def b5_lack(state: RecipeState) -> RecipeState:
    """B5 - stop when no compatible ingredient is available."""
    return {
        "error": "Error: no ingredient compatible with the declared dietary "
        "restrictions is available in the query or in the pantry."
    }


def b7_generate(state: RecipeState) -> RecipeState:
    """B7 - turn the compatible ingredients into a complete recipe."""
    identified = state.get("ingredients") or []
    search_results = state.get("search_results") or []

    llm = _get_llm()
    prompt = (
        "You are a professional chef. Create ONE complete personalized recipe.\n"
        "Constraints:\n"
        "- Respect ALL dietary restrictions declared in the query. No "
        "ingredient may violate them.\n"
        "- The recipe MUST use at least one of these available ingredients: "
        f"{json.dumps(identified)}.\n"
        "- Take the user's preferences into account.\n"
        "- Use the reference recipes below ONLY as a preparation reference "
        "(not as a source of base ingredients).\n\n"
        f"User query: {state['query']}\n"
        f"Reference recipes: {json.dumps(search_results)}\n\n"
        "Respond ONLY with a JSON object with ALL of these non-empty fields:\n"
        '{"Title": "...", "Ingredients": ["ingredient with quantity"], '
        '"Equipment": ["..."], "Instructions": ["step"], "Storage": "..."}'
    )
    try:
        raw = llm.invoke(prompt).content
        data = _parse_json(raw)
        return {"response": _format_recipe(data)}
    except Exception as exc:  # B6
        return {"error": f"Error: failed to generate the recipe ({exc})."}


# --------------------------------------------------------------------------- #
# Helpers                                                                      #
# --------------------------------------------------------------------------- #
def _parse_json(raw: str) -> dict:
    raw = (raw or "").strip()
    if raw.startswith("```"):
        raw = raw.split("```", 2)[1]
        if raw.startswith("json"):
            raw = raw[4:]
        raw = raw.strip()
    start = raw.find("{")
    end = raw.rfind("}")
    if start != -1 and end != -1:
        raw = raw[start : end + 1]
    return json.loads(raw)


def _format_recipe(data: dict) -> str:
    def as_lines(value) -> str:
        if isinstance(value, list):
            return "\n".join(f"- {item}" for item in value)
        return str(value)

    return (
        f"Title: {data.get('Title', '')}\n\n"
        f"Ingredients:\n{as_lines(data.get('Ingredients', []))}\n\n"
        f"Equipment:\n{as_lines(data.get('Equipment', []))}\n\n"
        f"Instructions:\n{as_lines(data.get('Instructions', []))}\n\n"
        f"Storage:\n{as_lines(data.get('Storage', ''))}"
    )


# --------------------------------------------------------------------------- #
# Routing                                                                      #
# --------------------------------------------------------------------------- #
def _route_after_b1(state: RecipeState) -> str:
    return "b2" if state.get("valid") else END


def _route_after_b2(state: RecipeState) -> str:
    return END if state.get("error") else "b3"


def _route_after_b3(state: RecipeState) -> str:
    if state.get("error"):
        return END
    return "b4" if state.get("ingredients") else "b5"


def _route_after_b4(state: RecipeState) -> str:
    return END if state.get("error") else "b7"


# --------------------------------------------------------------------------- #
# Graph assembly                                                               #
# --------------------------------------------------------------------------- #
def _build_graph():
    graph = StateGraph(RecipeState)
    graph.add_node("b1", b1_validate)
    graph.add_node("b2", b2_pantry)
    graph.add_node("b3", b3_identify)
    graph.add_node("b4", b4_search)
    graph.add_node("b5", b5_lack)
    graph.add_node("b7", b7_generate)

    graph.set_entry_point("b1")
    graph.add_conditional_edges("b1", _route_after_b1, {"b2": "b2", END: END})
    graph.add_conditional_edges("b2", _route_after_b2, {"b3": "b3", END: END})
    graph.add_conditional_edges("b3", _route_after_b3, {"b4": "b4", "b5": "b5", END: END})
    graph.add_conditional_edges("b4", _route_after_b4, {"b7": "b7", END: END})
    graph.add_edge("b5", END)
    graph.add_edge("b7", END)
    return graph.compile()


_APP = None


def _app():
    global _APP
    if _APP is None:
        _APP = _build_graph()
    return _APP


# --------------------------------------------------------------------------- #
# Entrypoint                                                                   #
# --------------------------------------------------------------------------- #
def agent(query: str) -> str:
    """Generate a personalized recipe for the given natural-language query."""
    try:
        final_state = _app().invoke({"query": query})
    except Exception as exc:  # B6 - never abort with an unhandled exception
        return f"Error: execution failure ({exc})."

    if final_state.get("error"):
        return final_state["error"]
    if final_state.get("response"):
        return final_state["response"]
    return "Error: the agent could not produce a recipe."


if __name__ == "__main__":
    import sys

    user_query = " ".join(sys.argv[1:]) or "I have chicken, rice and broccoli."
    print(agent(user_query))

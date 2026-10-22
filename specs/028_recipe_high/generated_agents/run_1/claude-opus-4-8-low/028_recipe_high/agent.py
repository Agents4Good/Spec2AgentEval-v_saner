"""Personalized recipe generator agent (high-complexity specification).

Persona: a chef that creates personalized recipes adapted to the ingredients
provided in the user's query or available in their pantry, while respecting
the user's declared dietary preferences and restrictions. It returns the
complete recipe with ingredients, equipment, instructions and storage
recommendations.

Model:      GPT-4o
Framework:  LangGraph
GenAI:      ChatOpenAI (langchain-openai)
Tools:      retrieve_pantry, search_recipes (both LangChain @tool, executed
            exclusively through their ``.invoke`` interface)
Entrypoint: agent(query: str) -> str

Behavioral graph:
    Q -> B1 (valid query?)
          invalid -> Error
          valid   -> B2 (retrieve_pantry, exactly once, first tool)
                       error   -> B6 (Error: execution failure)
                       success -> B3 (identify compatible ingredients)
                                    empty     -> B5 (Error: lack of ingredients)
                                    non-empty -> B4 (search_recipes, exactly once)
                                                   error -> B6
                                                   ok    -> B7 (generate recipe)
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
# Tools                                                                        #
# --------------------------------------------------------------------------- #
@tool
def retrieve_pantry() -> List[str]:
    """Obtain the ingredients available in the user's pantry from a Notion page.

    Takes no input. Returns the complete list of pantry ingredients (an empty
    pantry returns an empty list). On missing credentials, an inaccessible
    page, or a timeout, returns a string starting with ``Error:`` followed by
    an explanation of the failure.
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
            request_url = base_url
            if next_cursor:
                request_url = f"{base_url}&start_cursor={next_cursor}"
            resp = requests.get(request_url, headers=headers, timeout=30)
            if resp.status_code != 200:
                return (
                    f"Error: could not access the Notion pantry page "
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
    """Search the web (via Tavily) for recipes as a preparation reference.

    Input is the list of identified ingredients compatible with the dietary
    restrictions. Returns up to 5 results, each a dict with ``title``, ``url``
    and ``content`` (excerpt). No results returns an empty list. The recipes
    found are neither a source of base ingredients nor the final recipe. On
    missing credentials, an inaccessible page, or a timeout, returns a string
    starting with ``Error:`` followed by an explanation of the failure.
    """
    api_key = os.environ.get("TAVILY_API_KEY")
    if not api_key:
        return "Error: missing TAVILY_API_KEY credential to search for recipes."

    if not ingredients:
        return []

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
                f"Error: could not reach the recipe search service "
                f"(status {resp.status_code})."
            )
        data = resp.json()
        results = []
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
# LLM                                                                          #
# --------------------------------------------------------------------------- #
def _get_llm() -> ChatOpenAI:
    return ChatOpenAI(model="gpt-4o", temperature=0)


def _is_error(value) -> bool:
    return isinstance(value, str) and (
        value.lower().startswith("error:") or value.lower().startswith("erro:")
    )


def _normalize_error(value: str) -> str:
    """Return a message that always begins with ``Error:``."""
    text = str(value).strip()
    lowered = text.lower()
    if lowered.startswith("error:"):
        return "Error:" + text[len("error:"):]
    if lowered.startswith("erro:"):
        return "Error:" + text[len("erro:"):]
    return f"Error: {text}"


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
# Nodes                                                                        #
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
        "Respond with a JSON object: "
        '{"valid": true|false, "reason": "short explanation"}.\n\n'
        f"Query: {query}"
    )
    try:
        raw = llm.invoke(prompt).content
        data = _parse_json(raw)
        valid = bool(data.get("valid"))
        if valid:
            return {"valid": True}
        reason = data.get("reason") or "the query is not related to food/cooking."
        return {"valid": False, "error": f"Error: invalid query - {reason}"}
    except Exception as exc:
        return {"valid": False, "error": f"Error: failed to validate the query ({exc})."}


def b2_pantry(state: RecipeState) -> RecipeState:
    """B2 - retrieve the pantry ingredients (first tool, called exactly once)."""
    try:
        pantry = retrieve_pantry.invoke({})
    except Exception as exc:  # B6 - execution failure
        return {"error": f"Error: failed to retrieve the pantry ({exc})."}
    if _is_error(pantry):  # B6 - propagate tool failure
        return {"error": _normalize_error(pantry)}
    pantry = [str(i).strip() for i in (pantry or []) if str(i).strip()]
    return {"pantry_ingredients": pantry}


def b3_identify_ingredients(state: RecipeState) -> RecipeState:
    """B3 - identify usable ingredients compatible with dietary restrictions."""
    llm = _get_llm()
    prompt = (
        "Identify the usable ingredients for a recipe. The usable ingredients "
        "are those mentioned in the user's query PLUS all the ingredients "
        "returned from the pantry. Then keep only those compatible with the "
        "dietary restrictions declared in the query.\n"
        "Rules:\n"
        "- Consider regional variants and inflected/plural forms of the "
        "ingredients.\n"
        "- EXCLUDE any ingredient mentioned under negation, absence, or "
        "hypothesis (e.g. 'without X', 'I ran out of X', 'if I had X'). "
        "These are not available ingredients.\n"
        "- A declared dietary restriction implies ALL of its derived "
        "ingredients, even ones the user did not mention. EXCLUDE every "
        "ingredient incompatible with the declared dietary restrictions.\n"
        "Respond with a JSON object: "
        '{"ingredients": ["..."]}. If no ingredient is identified or '
        "compatible, return an empty list.\n\n"
        f"Query: {state['query']}\n"
        f"Pantry ingredients: {json.dumps(state.get('pantry_ingredients', []))}"
    )
    try:
        raw = llm.invoke(prompt).content
        data = _parse_json(raw)
        ingredients = data.get("ingredients") or []
        ingredients = [str(i).strip() for i in ingredients if str(i).strip()]
        return {"ingredients": ingredients}
    except Exception as exc:
        return {"error": f"Error: failed to identify ingredients ({exc})."}


def b4_search(state: RecipeState) -> RecipeState:
    """B4 - search reference recipes for the identified ingredients (once)."""
    try:
        results = search_recipes.invoke({"ingredients": state["ingredients"]})
    except Exception as exc:  # B6 - execution failure
        return {"error": f"Error: failed to search for recipes ({exc})."}
    if _is_error(results):  # B6 - propagate tool failure
        return {"error": _normalize_error(results)}
    return {"search_results": results}


def b5_lack(state: RecipeState) -> RecipeState:
    """B5 - stop when no compatible ingredient is available."""
    return {
        "error": "Error: no ingredient compatible with the declared dietary "
        "restrictions is available in the query or in the pantry."
    }


def b7_generate(state: RecipeState) -> RecipeState:
    """B7 - turn the compatible ingredients into a complete recipe."""
    ingredients = state.get("ingredients") or []
    search_results = state.get("search_results") or []

    llm = _get_llm()
    prompt = (
        "You are a professional chef. Create ONE complete personalized recipe "
        "using your culinary knowledge.\n"
        "Constraints:\n"
        "- Respect all dietary restrictions declared in the query. No "
        "ingredient may violate them.\n"
        f"- The recipe MUST use at least one of these identified ingredients: "
        f"{json.dumps(ingredients)}.\n"
        "- Take the user's preferences into account.\n"
        "- Use the reference recipes below only as a preparation reference "
        "(they are not a source of base ingredients nor the final recipe).\n\n"
        f"User query: {state['query']}\n"
        f"Reference recipes: {json.dumps(search_results)}\n\n"
        "Respond with a JSON object with ALL of these non-empty fields:\n"
        '{"Title": "...", "Ingredients": ["ingredient with quantity"], '
        '"Equipment": ["..."], "Instructions": ["step"], "Storage": "..."}'
    )
    try:
        raw = llm.invoke(prompt).content
        data = _parse_json(raw)
        response = _format_recipe(data)
        return {"response": response}
    except Exception as exc:  # B6 - model failure
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
    graph.add_node("b3", b3_identify_ingredients)
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

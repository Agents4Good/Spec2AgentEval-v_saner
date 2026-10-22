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

MODEL_NAME = "gpt-4o"
HTTP_TIMEOUT = 20
NOTION_VERSION = "2022-06-28"

PERSONA = (
    "You are an experienced chef whose role is to create personalized recipes "
    "adapted to the ingredients available to the user and respecting their "
    "dietary preferences and restrictions."
)


# --------------------------------------------------------------------------- #
# Tools
# --------------------------------------------------------------------------- #

def _rich_text_to_str(rich_text) -> str:
    return "".join(part.get("plain_text", "") for part in rich_text or []).strip()


@tool
def retrieve_pantry() -> list:
    """Retrieve the list of ingredients available in the user's pantry from a Notion page.

    Returns a list of ingredient names (empty if the pantry is empty) or a
    string starting with 'Error:' on failure.
    """
    api_key = os.environ.get("NOTION_API_KEY")
    page_id = os.environ.get("NOTION_PAGE_ID")
    if not api_key or not page_id:
        return "Error: missing Notion credentials (NOTION_API_KEY / NOTION_PAGE_ID)."
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Notion-Version": NOTION_VERSION,
    }
    url = f"https://api.notion.com/v1/blocks/{page_id}/children"
    ingredients: List[str] = []
    cursor = None
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
                btype = block.get("type")
                content = block.get(btype, {}) if btype else {}
                if not isinstance(content, dict) or "rich_text" not in content:
                    continue
                text = _rich_text_to_str(content.get("rich_text"))
                if not text:
                    continue
                for line in text.splitlines():
                    for item in line.split(","):
                        item = item.strip(" -•*\t")
                        if item:
                            ingredients.append(item)
            if not data.get("has_more"):
                break
            cursor = data.get("next_cursor")
    except requests.Timeout:
        return "Error: timeout while accessing the Notion page."
    except requests.RequestException as exc:
        return f"Error: failed to access the Notion page: {exc}"
    except ValueError as exc:
        return f"Error: invalid response from Notion: {exc}"
    return ingredients


@tool
def search_recipes(ingredients: List[str]) -> list:
    """Search the web (Tavily) for reference recipes using the given ingredients.

    Returns up to 5 results with title, url and content, an empty list if none,
    or a string starting with 'Error:' on failure.
    """
    api_key = os.environ.get("TAVILY_API_KEY")
    if not api_key:
        return "Error: missing Tavily credentials (TAVILY_API_KEY)."
    query = "recipe with " + ", ".join(ingredients)
    try:
        resp = requests.post(
            "https://api.tavily.com/search",
            json={"api_key": api_key, "query": query, "max_results": 5, "search_depth": "basic"},
            headers={"Authorization": f"Bearer {api_key}"},
            timeout=HTTP_TIMEOUT,
        )
        if resp.status_code != 200:
            return f"Error: Tavily search failed (HTTP {resp.status_code}): {resp.text[:200]}"
        results = resp.json().get("results", []) or []
    except requests.Timeout:
        return "Error: timeout while searching recipes on Tavily."
    except requests.RequestException as exc:
        return f"Error: failed to access Tavily: {exc}"
    except ValueError as exc:
        return f"Error: invalid response from Tavily: {exc}"
    return [
        {"title": r.get("title", ""), "url": r.get("url", ""), "content": r.get("content", "")}
        for r in results[:5]
    ]


# --------------------------------------------------------------------------- #
# Structured outputs
# --------------------------------------------------------------------------- #

class QueryValidation(BaseModel):
    is_valid: bool = Field(description="True if the query is a valid food/cooking request.")
    reason: str = Field(description="Short explanation of the classification.")


class IngredientIdentification(BaseModel):
    dietary_restrictions: List[str] = Field(default_factory=list)
    ingredients: List[str] = Field(
        default_factory=list,
        description="Usable ingredients compatible with the dietary restrictions.",
    )


class Recipe(BaseModel):
    title: str
    ingredients: List[str] = Field(description="Ingredients with quantities.")
    equipment: List[str]
    instructions: List[str] = Field(description="Step-by-step instructions.")
    storage: str


# --------------------------------------------------------------------------- #
# Graph
# --------------------------------------------------------------------------- #

class AgentState(TypedDict, total=False):
    query: str
    pantry: list
    ingredients: list
    references: list
    result: str
    error: Optional[str]


def _llm() -> ChatOpenAI:
    return ChatOpenAI(model=MODEL_NAME, temperature=0)


def _is_error(value) -> bool:
    return isinstance(value, str) and value.strip().lower().startswith(("error:", "erro:"))


def _as_error(msg: str) -> str:
    msg = str(msg).strip()
    if msg.lower().startswith("erro:"):
        msg = msg[5:].strip()
    elif msg.lower().startswith("error:"):
        msg = msg[6:].strip()
    return f"Error: {msg}"


def validate_node(state: AgentState) -> AgentState:
    query = (state.get("query") or "").strip()
    if not query:
        return {"error": "Error: invalid query - the query is empty."}
    prompt = (
        "Classify the user query. It is INVALID if it is empty/meaningless, offensive, "
        "or unrelated to food/cooking. Otherwise it is VALID (any request about a recipe, "
        "meal, dish, ingredients, or cooking, even if vague)."
    )
    res = _llm().with_structured_output(QueryValidation).invoke(
        [SystemMessage(content=prompt), HumanMessage(content=query)]
    )
    if not res.is_valid:
        return {"error": f"Error: invalid query - {res.reason}"}
    return {}


def pantry_node(state: AgentState) -> AgentState:
    result = retrieve_pantry.invoke({})
    if _is_error(result):
        return {"error": _as_error(result)}
    if not isinstance(result, list):
        return {"error": "Error: unexpected response from retrieve_pantry."}
    return {"pantry": result}


def identify_node(state: AgentState) -> AgentState:
    prompt = (
        PERSONA + "\n\nIdentify the ingredients usable in the recipe. Rules:\n"
        "- Usable ingredients = ingredients explicitly stated as available/desired in the "
        "user query PLUS ALL pantry ingredients.\n"
        "- Exclude ingredients mentioned under negation, absence or hypothesis (e.g. "
        "'I don't have eggs', 'no milk', 'if I had cheese', 'maybe I could buy chicken').\n"
        "- Determine the dietary restrictions declared in the query (e.g. vegan, vegetarian, "
        "gluten-free, lactose-free, allergies, dislikes). A restriction implies all of its "
        "derived ingredients even if not mentioned (e.g. vegan excludes meat, fish, eggs, "
        "dairy, honey; gluten-free excludes wheat flour, bread, pasta, barley, rye; "
        "lactose-free excludes milk, butter, cheese, cream, yogurt).\n"
        "- Keep only ingredients compatible with all restrictions.\n"
        "- Treat regional variants and inflected forms as the same ingredient (no duplicates).\n"
        "- If nothing remains, return an empty list."
    )
    content = (
        f"User query: {state['query']}\n\n"
        f"Pantry ingredients: {json.dumps(state.get('pantry', []), ensure_ascii=False)}"
    )
    res = _llm().with_structured_output(IngredientIdentification).invoke(
        [SystemMessage(content=prompt), HumanMessage(content=content)]
    )
    ingredients = [i.strip() for i in res.ingredients if i and i.strip()]
    if not ingredients:
        return {
            "ingredients": [],
            "error": "Error: no ingredients compatible with the dietary restrictions were "
            "found in the query or in the pantry.",
        }
    return {"ingredients": ingredients}


def search_node(state: AgentState) -> AgentState:
    result = search_recipes.invoke({"ingredients": state["ingredients"]})
    if _is_error(result):
        return {"error": _as_error(result)}
    if not isinstance(result, list):
        return {"error": "Error: unexpected response from search_recipes."}
    return {"references": result}


def generate_node(state: AgentState) -> AgentState:
    prompt = (
        PERSONA + "\n\nCreate ONE complete recipe using your culinary knowledge.\n"
        "- Use primarily the available ingredients listed; include at least one of them.\n"
        "- You may assume basic staples (water, salt) only if compatible with the restrictions.\n"
        "- NEVER include any ingredient that violates the user's dietary restrictions, "
        "including derived ingredients.\n"
        "- Respect the user's preferences in the query.\n"
        "- Reference recipes are only preparation inspiration, not a source of ingredients.\n"
        "- Give quantities for each ingredient, equipment list, step-by-step instructions "
        "and storage recommendations. Answer in the language of the user's query."
    )
    refs = state.get("references") or []
    content = (
        f"User query: {state['query']}\n\n"
        f"Available compatible ingredients: {json.dumps(state['ingredients'], ensure_ascii=False)}\n\n"
        f"Reference recipes: {json.dumps(refs, ensure_ascii=False) if refs else 'none'}"
    )
    r = _llm().with_structured_output(Recipe).invoke(
        [SystemMessage(content=prompt), HumanMessage(content=content)]
    )
    if not (r.title.strip() and r.ingredients and r.equipment and r.instructions and r.storage.strip()):
        return {"error": "Error: the generated recipe is incomplete."}
    text = (
        f"Title: {r.title.strip()}\n\n"
        "Ingredients:\n" + "\n".join(f"- {i}" for i in r.ingredients) + "\n\n"
        "Equipment:\n" + "\n".join(f"- {e}" for e in r.equipment) + "\n\n"
        "Instructions:\n" + "\n".join(f"{n}. {s}" for n, s in enumerate(r.instructions, 1)) + "\n\n"
        f"Storage: {r.storage.strip()}"
    )
    return {"result": text}


def _route(next_node: str):
    def router(state: AgentState) -> str:
        return END if state.get("error") else next_node
    return router


def build_graph():
    g = StateGraph(AgentState)
    g.add_node("validate", validate_node)
    g.add_node("pantry", pantry_node)
    g.add_node("identify", identify_node)
    g.add_node("search", search_node)
    g.add_node("generate", generate_node)
    g.set_entry_point("validate")
    g.add_conditional_edges("validate", _route("pantry"), ["pantry", END])
    g.add_conditional_edges("pantry", _route("identify"), ["identify", END])
    g.add_conditional_edges("identify", _route("search"), ["search", END])
    g.add_conditional_edges("search", _route("generate"), ["generate", END])
    g.add_edge("generate", END)
    return g.compile()


def agent(query: str) -> str:
    """Entrypoint: returns a complete recipe or a message starting with 'Error:'."""
    try:
        final = build_graph().invoke({"query": query if isinstance(query, str) else ""})
    except Exception as exc:  # B6: never raise
        return f"Error: execution failure - {type(exc).__name__}: {exc}"
    if final.get("error"):
        return final["error"]
    return final.get("result") or "Error: no recipe was generated."


if __name__ == "__main__":
    import sys
    print(agent(" ".join(sys.argv[1:]) or "A vegan dinner with tomatoes and rice"))

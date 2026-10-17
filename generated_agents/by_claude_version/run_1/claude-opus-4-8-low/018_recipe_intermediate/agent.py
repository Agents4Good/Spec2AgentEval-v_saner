"""
Personalized Recipe Generator Agent
====================================

A LangGraph agent that assumes the persona of a *chef* and generates
personalized recipes, using either the ingredients mentioned in the user's
query or the ingredients available in the user's pantry (a Notion page), while
respecting the user's dietary preferences and restrictions.

Interface contract (benchmark)
------------------------------
* Model      : gpt-4o
* Framework  : LangGraph
* GenAI Tool : ChatOpenAI
* Tools      : LangChain ``@tool`` objects, invoked exclusively through their
               ``.invoke`` interface (never by calling the implementation
               function directly).
* Entrypoint : ``agent(query: str) -> dict``
* Output     : the final graph state (``dict``). The user-facing answer lives in
               ``state["output"]``: either the formatted recipe or a string that
               starts with ``"Error:"``.

Environment variables
---------------------
* ``OPENAI_API_KEY``  (required)
* ``TAVILY_API_KEY``  (required only when ``search_recipes`` is used)
* ``NOTION_API_KEY``  (required only when ``retrieve_pantry`` is used)
* ``NOTION_PAGE_ID``  (required only when ``retrieve_pantry`` is used)
* ``OPENAI_MODEL``            (optional, default ``gpt-4o``)
* ``REQUEST_TIMEOUT_SECONDS`` (optional, default ``20``)
"""

from __future__ import annotations

import os
from typing import Any, Dict, List, Literal, Optional, TypedDict

import httpx
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langgraph.graph import END, StateGraph
from pydantic import BaseModel, Field

load_dotenv()

# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #

DEFAULT_MODEL = "gpt-4o"
NOTION_API_VERSION = "2022-06-28"
NOTION_BLOCKS_URL = "https://api.notion.com/v1/blocks/{block_id}/children"
TAVILY_SEARCH_URL = "https://api.tavily.com/search"

ERROR_PREFIX = "Error:"          # agent-facing error prefix (B1, B5, B6)
TOOL_ERROR_PREFIX = "erro:"      # search_recipes failure prefix (spec)
PANTRY_ERROR_PREFIX = "Error:"   # retrieve_pantry failure prefix (spec)


def _request_timeout() -> float:
    try:
        return float(os.getenv("REQUEST_TIMEOUT_SECONDS", "20"))
    except ValueError:
        return 20.0


def get_llm() -> ChatOpenAI:
    """Builds the ChatOpenAI client used by every LLM node (patchable in tests)."""
    return ChatOpenAI(
        model=os.getenv("OPENAI_MODEL", DEFAULT_MODEL),
        temperature=0,
        timeout=_request_timeout() * 2,
        max_retries=1,
    )


# --------------------------------------------------------------------------- #
# Persona
# --------------------------------------------------------------------------- #

CHEF_PERSONA = (
    "You are an experienced, friendly professional chef whose job is to create "
    "personalized recipes. You are always respectful and polite and never use "
    "offensive or discriminatory language. You take dietary restrictions "
    "extremely seriously: a dietary restriction implies ALL of its derived "
    "ingredients, even ones the user did not mention explicitly, and you never "
    "suggest an ingredient that violates a restriction."
)


# --------------------------------------------------------------------------- #
# Structured-output schemas (LLM -> Python)
# --------------------------------------------------------------------------- #


class QueryValidation(BaseModel):
    """B1: is the query acceptable for recipe generation?"""

    is_food_related: bool = Field(
        description="True if the query is about food, cooking, recipes, meals, ingredients or diets."
    )
    is_offensive: bool = Field(
        description="True if the query contains offensive, hateful, harassing or abusive language."
    )
    reason: str = Field(description="Short, polite explanation of the decision.")


class QueryAnalysis(BaseModel):
    """B2: ingredients and dietary information extracted from the query."""

    ingredients: List[str] = Field(
        default_factory=list,
        description=(
            "Ingredients that are genuinely AVAILABLE according to the query. "
            "Normalize regional variants and inflected forms to a canonical name. "
            "EXCLUDE anything mentioned under negation, absence or hypothesis "
            "(e.g. 'without onion', 'I ran out of eggs', 'if I had milk'). "
            "EXCLUDE ingredients that are incompatible with the dietary "
            "restrictions. Empty list if none."
        ),
    )
    dietary_restrictions: List[str] = Field(
        default_factory=list,
        description="Dietary restrictions (e.g. vegan, gluten-free, nut allergy, lactose intolerant).",
    )
    preferences: List[str] = Field(
        default_factory=list,
        description="Other preferences (cuisine, meal type, time available, spice level, etc.).",
    )


class Recipe(BaseModel):
    title: str = Field(description="Recipe title.")
    ingredients: List[str] = Field(description="Ingredients with quantities, one per item.")
    equipment: List[str] = Field(description="Kitchen equipment needed.")
    instructions: List[str] = Field(description="Step-by-step preparation instructions.")
    storage: str = Field(description="Storage recommendations (fridge/freezer, duration, reheating).")


class CompatibilityFilter(BaseModel):
    """Keeps only pantry ingredients compatible with the dietary restrictions (B4)."""

    compatible: List[str] = Field(
        default_factory=list,
        description="Pantry ingredients that respect ALL declared dietary restrictions.",
    )


# --------------------------------------------------------------------------- #
# Tools
# --------------------------------------------------------------------------- #


def _extract_block_text(block: Dict[str, Any]) -> List[str]:
    """Extracts plain text from the Notion block types that can hold pantry items."""
    block_type = block.get("type", "")
    payload = block.get(block_type, {}) or {}

    if block_type == "table_row":
        texts = []
        for cell in payload.get("cells", []):
            text = "".join(rt.get("plain_text", "") for rt in cell).strip()
            if text:
                texts.append(text)
        return [" - ".join(texts)] if texts else []

    rich_text = payload.get("rich_text")
    if rich_text is None:
        return []
    text = "".join(rt.get("plain_text", "") for rt in rich_text).strip()
    if not text:
        return []
    if block_type in {"paragraph", "quote", "callout"} and "," in text:
        return [part.strip() for part in text.split(",") if part.strip()]
    return [text]


def _notion_pantry() -> List[str]:
    """Reads the pantry ingredients from the configured Notion page (HTTP layer)."""
    api_key = os.getenv("NOTION_API_KEY", "").strip()
    page_id = os.getenv("NOTION_PAGE_ID", "").strip()
    if not api_key:
        raise RuntimeError("missing NOTION_API_KEY")
    if not page_id:
        raise RuntimeError("missing NOTION_PAGE_ID")

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Notion-Version": NOTION_API_VERSION,
    }
    ingredients: List[str] = []
    cursor: Optional[str] = None

    with httpx.Client(timeout=_request_timeout()) as client:
        while True:
            params: Dict[str, Any] = {"page_size": 100}
            if cursor:
                params["start_cursor"] = cursor
            resp = client.get(
                NOTION_BLOCKS_URL.format(block_id=page_id), headers=headers, params=params
            )
            resp.raise_for_status()
            data = resp.json()
            for block in data.get("results", []):
                ingredients.extend(_extract_block_text(block))
            if not data.get("has_more"):
                break
            cursor = data.get("next_cursor")

    seen = set()
    unique = []
    for item in ingredients:
        key = item.lower()
        if key and key not in seen:
            seen.add(key)
            unique.append(item)
    return unique


def _tavily_search(ingredients: List[str]) -> List[Dict[str, str]]:
    """Searches recipes on Tavily for the given ingredients (HTTP layer)."""
    api_key = os.getenv("TAVILY_API_KEY", "").strip()
    if not api_key:
        raise RuntimeError("missing TAVILY_API_KEY")

    query = "recipe with " + ", ".join(ingredients)
    payload = {
        "api_key": api_key,
        "query": query,
        "search_depth": "basic",
        "max_results": 5,
        "include_answer": False,
    }
    with httpx.Client(timeout=_request_timeout()) as client:
        resp = client.post(TAVILY_SEARCH_URL, json=payload)
        resp.raise_for_status()
        data = resp.json()

    results = []
    for item in data.get("results", [])[:5]:
        results.append(
            {
                "title": str(item.get("title", "")).strip(),
                "url": str(item.get("url", "")).strip(),
                "content": str(item.get("content", "")).strip()[:1500],
            }
        )
    return results


def _describe_exception(exc: Exception, context: str) -> str:
    if isinstance(exc, (httpx.TimeoutException, TimeoutError)):
        return f"{context} timed out. Please try again later."
    if isinstance(exc, httpx.HTTPStatusError):
        return f"{context} failed: the page is inaccessible (HTTP {exc.response.status_code})."
    text = str(exc).strip() or exc.__class__.__name__
    if "missing" in text.lower() or "api key" in text.lower() or "api_key" in text.lower():
        return f"{context} failed due to missing credentials ({text})."
    return f"{context} failed: {text}"


@tool
def retrieve_pantry() -> Any:
    """Retrieves the user's pantry (list of available ingredients) from the user's Notion page.

    Returns the complete list of pantry ingredients (an empty list when the
    pantry is empty). On failure (missing credentials, inaccessible page or
    timeout) returns a string beginning with ``Error:`` explaining the problem.
    """
    try:
        return _notion_pantry()
    except Exception as exc:  # noqa: BLE001 - surfaced as an Error: message
        return f"{PANTRY_ERROR_PREFIX} {_describe_exception(exc, 'pantry retrieval')}"


@tool
def search_recipes(ingredients: List[str]) -> Any:
    """Searches the web (Tavily) for recipes that serve as a preparation reference.

    The returned recipes are neither a source of base ingredients nor the final
    recipe. Returns up to 5 results, each with ``title``, ``url`` and
    ``content`` (an empty list when there are no results). On failure (missing
    credentials, inaccessible page or timeout) returns a string beginning with
    ``erro:`` explaining the problem.
    """
    try:
        return _tavily_search(list(ingredients or []))
    except Exception as exc:  # noqa: BLE001 - surfaced as an erro: message
        return f"{TOOL_ERROR_PREFIX} {_describe_exception(exc, 'recipe search')}"


TOOLS = [retrieve_pantry, search_recipes]


# --------------------------------------------------------------------------- #
# Graph state
# --------------------------------------------------------------------------- #


class AgentState(TypedDict, total=False):
    query: str
    is_valid: bool
    query_ingredients: List[str]
    dietary_restrictions: List[str]
    preferences: List[str]
    tool_called: Optional[Literal["search_recipes", "retrieve_pantry"]]
    search_results: List[Dict[str, str]]
    base_ingredients: List[str]
    recipe: Optional[Dict[str, Any]]
    error: Optional[str]
    output: str


def _fail(message: str) -> Dict[str, Any]:
    """Produces the error state update (``Error: [clear explanation]``)."""
    message = message.strip()
    if not message.startswith(ERROR_PREFIX):
        message = f"{ERROR_PREFIX} {message}"
    return {"error": message, "output": message}


def _is_tool_error(value: Any) -> bool:
    """True when a tool returned a failure string (``Error:`` or ``erro:``)."""
    return isinstance(value, str) and (
        value.lstrip().lower().startswith("error:")
        or value.lstrip().lower().startswith("erro:")
    )


# --------------------------------------------------------------------------- #
# Nodes
# --------------------------------------------------------------------------- #


def validate_query(state: AgentState) -> Dict[str, Any]:
    """B1: reject empty, offensive or non-food-related queries."""
    query = (state.get("query") or "").strip()
    if not query:
        return {
            "is_valid": False,
            **_fail("the query is empty. Please describe the dish, ingredients or preferences you have."),
        }

    try:
        llm = get_llm().with_structured_output(QueryValidation)
        verdict: QueryValidation = llm.invoke(
            [
                SystemMessage(
                    content=CHEF_PERSONA
                    + "\n\nClassify the user's query. Decide whether it is related to "
                    "food/cooking/recipes/ingredients/diets and whether it contains offensive "
                    "language. Be strict: greetings, programming questions, math and politics "
                    "are NOT food related."
                ),
                HumanMessage(content=query),
            ]
        )
    except Exception as exc:  # B6
        return {"is_valid": False, **_fail(_describe_exception(exc, "query validation"))}

    if verdict.is_offensive:
        return {
            "is_valid": False,
            **_fail("the query contains offensive language. Please rephrase it respectfully."),
        }
    if not verdict.is_food_related:
        return {
            "is_valid": False,
            **_fail(f"the query is not related to food or cooking ({verdict.reason}). I can only generate recipes."),
        }
    return {"is_valid": True, "error": None}


def analyze_query(state: AgentState) -> Dict[str, Any]:
    """B2: extract compatible ingredients, dietary restrictions and preferences."""
    try:
        llm = get_llm().with_structured_output(QueryAnalysis)
        analysis: QueryAnalysis = llm.invoke(
            [
                SystemMessage(
                    content=CHEF_PERSONA
                    + "\n\nExtract from the user's query: (1) the ingredients that are "
                    "genuinely available (apply every exclusion rule in the field "
                    "description); (2) dietary restrictions; (3) other preferences. "
                    "Return empty lists when nothing applies."
                ),
                HumanMessage(content=state["query"]),
            ]
        )
    except Exception as exc:  # B6
        return _fail(_describe_exception(exc, "query analysis"))

    return {
        "query_ingredients": [i.strip() for i in analysis.ingredients if i.strip()],
        "dietary_restrictions": [r.strip() for r in analysis.dietary_restrictions if r.strip()],
        "preferences": [p.strip() for p in analysis.preferences if p.strip()],
    }


def call_search_recipes(state: AgentState) -> Dict[str, Any]:
    """B3: ingredients were identified -> search reference recipes on Tavily."""
    ingredients = state.get("query_ingredients") or []
    results = search_recipes.invoke({"ingredients": ingredients})

    if _is_tool_error(results):  # B6
        return {"tool_called": "search_recipes", **_fail(_describe_exception(RuntimeError(str(results)), "recipe search"))}

    return {
        "tool_called": "search_recipes",
        "search_results": results if isinstance(results, list) else [],
        "base_ingredients": list(ingredients),
    }


def call_retrieve_pantry(state: AgentState) -> Dict[str, Any]:
    """B4: no ingredients in the query -> read the pantry and filter by restrictions."""
    pantry = retrieve_pantry.invoke({})

    if _is_tool_error(pantry):  # B6
        return {"tool_called": "retrieve_pantry", **_fail(_describe_exception(RuntimeError(str(pantry)), "pantry retrieval"))}

    pantry = [str(i).strip() for i in (pantry or []) if str(i).strip()]
    restrictions = state.get("dietary_restrictions") or []

    if pantry and restrictions:
        try:
            llm = get_llm().with_structured_output(CompatibilityFilter)
            filtered: CompatibilityFilter = llm.invoke(
                [
                    SystemMessage(
                        content=CHEF_PERSONA
                        + "\n\nGiven a list of pantry ingredients and the user's dietary "
                        "restrictions, keep ONLY the ingredients compatible with ALL of the "
                        "restrictions (remember a restriction implies its derived ingredients)."
                    ),
                    HumanMessage(
                        content=f"Dietary restrictions: {', '.join(restrictions)}\n"
                        f"Pantry ingredients: {', '.join(pantry)}"
                    ),
                ]
            )
            pantry = [i.strip() for i in filtered.compatible if i.strip()]
        except Exception as exc:  # B6
            return {"tool_called": "retrieve_pantry", **_fail(_describe_exception(exc, "pantry filtering"))}

    return {"tool_called": "retrieve_pantry", "base_ingredients": pantry}


def check_pantry_ingredients(state: AgentState) -> Dict[str, Any]:
    """B5: the pantry provided no compatible ingredient -> stop."""
    if state.get("base_ingredients"):
        return {}
    return _fail(
        "no ingredients compatible with your dietary restrictions were found in your "
        "pantry, so I cannot generate a recipe."
    )


def _format_search_context(state: AgentState) -> str:
    results = state.get("search_results") or []
    if not results:
        return "No reference recipes were returned by the search; rely on the identified ingredients."
    lines = ["Reference recipes found on the web (use ONLY as preparation inspiration):"]
    for i, r in enumerate(results, 1):
        lines.append(f"{i}. {r.get('title', '')} ({r.get('url', '')})\n   {r.get('content', '')}")
    return "\n".join(lines)


def generate_recipe(state: AgentState) -> Dict[str, Any]:
    """B7: turn the compatible ingredients into a complete recipe."""
    restrictions = state.get("dietary_restrictions") or []
    preferences = state.get("preferences") or []
    base_ingredients = state.get("base_ingredients") or []

    system = (
        CHEF_PERSONA
        + "\n\nCreate ONE complete recipe following these rules:\n"
        "1. The recipe MUST use the base ingredients listed below (from the query or the pantry) "
        "and MUST contain at least one of them.\n"
        "2. You may add only minimal essential extras (salt, oil, water).\n"
        "3. The recipe MUST respect ALL dietary restrictions; never include a violating "
        "ingredient, not even as optional.\n"
        "4. Follow the user's preferences whenever possible.\n"
        "5. Provide: a title, ingredients with quantities, required equipment, step-by-step "
        "instructions and storage recommendations. Every field must be non-empty.\n"
        "6. Write the recipe in the same language as the user's query.\n"
        "7. The web reference recipes are inspiration only, never a source of base ingredients."
    )
    human = (
        f"User query: {state['query']}\n\n"
        f"Base ingredients: {', '.join(base_ingredients)}\n"
        f"Dietary restrictions: {', '.join(restrictions) if restrictions else 'none'}\n"
        f"Preferences: {', '.join(preferences) if preferences else 'none'}\n\n"
    )
    if state.get("tool_called") == "search_recipes":
        human += _format_search_context(state)
    else:
        human += "Ingredients come from the user's pantry."

    try:
        llm = get_llm().with_structured_output(Recipe)
        recipe: Recipe = llm.invoke([SystemMessage(content=system), HumanMessage(content=human)])
    except Exception as exc:  # B6
        return _fail(_describe_exception(exc, "recipe generation"))

    return {"recipe": recipe.model_dump()}


def format_output(state: AgentState) -> Dict[str, Any]:
    """Renders the recipe in the format required by the specification."""
    return {"output": format_recipe(state["recipe"]), "error": None}


def format_recipe(recipe: Dict[str, Any]) -> str:
    lines: List[str] = []
    lines.append(f"Title: {recipe.get('title', '')}")
    lines.append("")
    lines.append("Ingredients:")
    lines.extend(f"- {item}" for item in recipe.get("ingredients", []))
    lines.append("")
    lines.append("Equipment:")
    lines.extend(f"- {item}" for item in recipe.get("equipment", []))
    lines.append("")
    lines.append("Instructions:")
    lines.extend(f"{i}. {step}" for i, step in enumerate(recipe.get("instructions", []), 1))
    lines.append("")
    lines.append(f"Storage: {recipe.get('storage', '')}")
    return "\n".join(lines)


# --------------------------------------------------------------------------- #
# Routing
# --------------------------------------------------------------------------- #


def _has_error(state: AgentState) -> bool:
    return bool(state.get("error"))


def route_after_validation(state: AgentState) -> str:
    return "error" if _has_error(state) or not state.get("is_valid") else "analyze_query"


def route_tool(state: AgentState) -> str:
    """B2 -> B3/B4: exactly one tool is chosen based on the presence of ingredients."""
    if _has_error(state):
        return "error"
    return "call_search_recipes" if state.get("query_ingredients") else "call_retrieve_pantry"


def route_after_search(state: AgentState) -> str:
    return "error" if _has_error(state) else "generate_recipe"


def route_after_pantry(state: AgentState) -> str:
    return "error" if _has_error(state) else "check_pantry_ingredients"


def route_after_check(state: AgentState) -> str:
    return "error" if _has_error(state) else "generate_recipe"


def route_after_generate(state: AgentState) -> str:
    return "error" if _has_error(state) else "format_output"


# --------------------------------------------------------------------------- #
# Graph
# --------------------------------------------------------------------------- #


def build_graph():
    """Builds and compiles the LangGraph workflow."""
    workflow = StateGraph(AgentState)

    workflow.add_node("validate_query", validate_query)
    workflow.add_node("analyze_query", analyze_query)
    workflow.add_node("call_search_recipes", call_search_recipes)
    workflow.add_node("call_retrieve_pantry", call_retrieve_pantry)
    workflow.add_node("check_pantry_ingredients", check_pantry_ingredients)
    workflow.add_node("generate_recipe", generate_recipe)
    workflow.add_node("format_output", format_output)

    workflow.set_entry_point("validate_query")
    workflow.add_conditional_edges(
        "validate_query", route_after_validation,
        {"analyze_query": "analyze_query", "error": END},
    )
    workflow.add_conditional_edges(
        "analyze_query", route_tool,
        {"call_search_recipes": "call_search_recipes",
         "call_retrieve_pantry": "call_retrieve_pantry", "error": END},
    )
    workflow.add_conditional_edges(
        "call_search_recipes", route_after_search,
        {"generate_recipe": "generate_recipe", "error": END},
    )
    workflow.add_conditional_edges(
        "call_retrieve_pantry", route_after_pantry,
        {"check_pantry_ingredients": "check_pantry_ingredients", "error": END},
    )
    workflow.add_conditional_edges(
        "check_pantry_ingredients", route_after_check,
        {"generate_recipe": "generate_recipe", "error": END},
    )
    workflow.add_conditional_edges(
        "generate_recipe", route_after_generate,
        {"format_output": "format_output", "error": END},
    )
    workflow.add_edge("format_output", END)

    return workflow.compile()


# --------------------------------------------------------------------------- #
# Entrypoint
# --------------------------------------------------------------------------- #


def _initial_state(query: Any) -> AgentState:
    return AgentState(
        query=query if isinstance(query, str) else ("" if query is None else str(query)),
        is_valid=False,
        query_ingredients=[],
        dietary_restrictions=[],
        preferences=[],
        tool_called=None,
        search_results=[],
        base_ingredients=[],
        recipe=None,
        error=None,
        output="",
    )


def agent(query: str) -> dict:
    """
    Entrypoint. Runs the recipe-generation workflow for ``query`` and returns the
    final state as a ``dict``.

    ``state["output"]`` holds either the formatted recipe or a string starting
    with ``"Error:"``. ``state["error"]`` is ``None`` on success.
    """
    state = _initial_state(query)

    if not os.getenv("OPENAI_API_KEY", "").strip():
        state.update(_fail("missing OPENAI_API_KEY; the language model cannot be reached."))
        return dict(state)

    try:
        graph = build_graph()
        final_state = graph.invoke(state)
    except Exception as exc:  # B6 - last line of defence
        state.update(_fail(_describe_exception(exc, "agent execution")))
        return dict(state)

    result = dict(final_state)
    if not result.get("output"):
        result.update(_fail("the agent finished without producing a recipe."))
    return result


if __name__ == "__main__":
    import sys

    user_query = " ".join(sys.argv[1:]) or "I want a vegan dinner with chickpeas and spinach"
    print(agent(user_query)["output"])

"""
Personalized Recipe Generator Agent
====================================

LangGraph agent that acts as a *chef* and generates personalized recipes,
using either the ingredients mentioned in the user's query or the ingredients
available in the user's pantry (a Notion page), while respecting the user's
dietary restrictions and preferences.

Interface contract (benchmark)
------------------------------
* Model      : gpt-4o
* Framework  : LangGraph
* GenAI Tool : ChatOpenAI
* Entrypoint : ``agent(query: str) -> dict``
* Output     : the final graph state (``dict``). The user-facing answer lives in
               ``state["output"]``: either the formatted recipe or a string that
               starts with ``"Error: "``.

Tools (generated together with the agent)
-----------------------------------------
* ``retrieve_pantry()``            -> reads the user's pantry from a Notion page.
* ``search_recipes(ingredients)``  -> searches recipes on Tavily for the given
                                      ingredients.

Environment variables
---------------------
* ``OPENAI_API_KEY``          (required)
* ``TAVILY_API_KEY``          (required only when ``search_recipes`` is used)
* ``NOTION_API_KEY``          (required only when ``retrieve_pantry`` is used)
* ``NOTION_PANTRY_PAGE_ID``   (required only when ``retrieve_pantry`` is used)
* ``OPENAI_MODEL``            (optional, default ``gpt-4o``)
* ``REQUEST_TIMEOUT_SECONDS`` (optional, default ``20``)
"""

from __future__ import annotations

import os
import re
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
MAX_COMPLIANCE_ATTEMPTS = 2  # generate -> verify -> (regenerate once) -> verify

ERROR_PREFIX = "Error: "


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
    "personalized recipes. You are always respectful and polite: you never use "
    "offensive, rude or discriminatory language, even if the user does. "
    "You take dietary restrictions extremely seriously and never suggest an "
    "ingredient that violates them."
)


# --------------------------------------------------------------------------- #
# Structured-output schemas (LLM -> Python)
# --------------------------------------------------------------------------- #


class QueryValidation(BaseModel):
    """Result of BR-01: is the query acceptable for recipe generation?"""

    is_food_related: bool = Field(
        description="True if the query is about food, cooking, recipes, meals, ingredients or diets."
    )
    is_offensive: bool = Field(
        description="True if the query contains offensive, hateful, harassing or abusive language."
    )
    reason: str = Field(description="Short, polite explanation of the decision.")


class QueryAnalysis(BaseModel):
    """Result of BR-02: what does the query tell us about ingredients and diet?"""

    ingredients: List[str] = Field(
        default_factory=list,
        description="Ingredients EXPLICITLY mentioned in the query (empty if none).",
    )
    dietary_restrictions: List[str] = Field(
        default_factory=list,
        description="Dietary restrictions (e.g. vegan, gluten-free, nut allergy, lactose intolerant).",
    )
    preferences: List[str] = Field(
        default_factory=list,
        description="Other preferences (cuisine, meal type, time available, spice level, etc.).",
    )


class NutritionInfo(BaseModel):
    calories: str = Field(description="Estimated calories per serving, e.g. '450 kcal'.")
    protein: str = Field(description="Protein per serving, e.g. '30 g'.")
    carbohydrates: str = Field(description="Carbohydrates per serving, e.g. '40 g'.")
    fat: str = Field(description="Fat per serving, e.g. '15 g'.")
    health_benefits: List[str] = Field(description="Main health benefits of the dish.")


class Recipe(BaseModel):
    title: str = Field(description="Recipe title.")
    servings: str = Field(description="Number of servings, e.g. '2 servings'.")
    ingredients: List[str] = Field(description="Ingredients with quantities, one per item.")
    equipment: List[str] = Field(description="Kitchen equipment needed.")
    instructions: List[str] = Field(description="Step-by-step preparation instructions.")
    nutrition: NutritionInfo
    storage: str = Field(description="Storage recommendations (fridge/freezer, duration, reheating).")


class ComplianceCheck(BaseModel):
    compliant: bool = Field(
        description="True only if every ingredient/instruction respects ALL dietary restrictions."
    )
    issues: List[str] = Field(
        default_factory=list, description="Concrete violations found (empty if compliant)."
    )


# --------------------------------------------------------------------------- #
# Tools
# --------------------------------------------------------------------------- #


def _notion_pantry() -> List[str]:
    """Reads the pantry ingredients from the configured Notion page (HTTP layer)."""
    api_key = os.getenv("NOTION_API_KEY", "").strip()
    page_id = os.getenv("NOTION_PANTRY_PAGE_ID", "").strip()
    if not api_key:
        raise RuntimeError("missing NOTION_API_KEY")
    if not page_id:
        raise RuntimeError("missing NOTION_PANTRY_PAGE_ID")

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
            resp = client.get(NOTION_BLOCKS_URL.format(block_id=page_id), headers=headers, params=params)
            resp.raise_for_status()
            data = resp.json()
            for block in data.get("results", []):
                ingredients.extend(_extract_block_text(block))
            if not data.get("has_more"):
                break
            cursor = data.get("next_cursor")

    # De-duplicate while preserving order.
    seen = set()
    unique = []
    for item in ingredients:
        key = item.lower()
        if key and key not in seen:
            seen.add(key)
            unique.append(item)
    return unique


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
    # A single line may contain several comma-separated items.
    if block_type in {"paragraph", "quote", "callout"} and "," in text:
        return [part.strip() for part in text.split(",") if part.strip()]
    return [text]


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
    for item in data.get("results", []):
        results.append(
            {
                "title": str(item.get("title", "")).strip(),
                "url": str(item.get("url", "")).strip(),
                "content": str(item.get("content", "")).strip()[:1500],
            }
        )
    return results


@tool
def retrieve_pantry() -> List[str]:
    """Retrieves the user's pantry (list of available ingredients) from the user's Notion page.
    Use it when the query is about food/cooking but does not mention any ingredient."""
    return _notion_pantry()


@tool
def search_recipes(ingredients: List[str]) -> List[Dict[str, str]]:
    """Searches the web (Tavily) for recipes matching the given list of ingredients.
    Returns a list of {title, url, content}. Use it when the query mentions ingredients explicitly."""
    return _tavily_search(ingredients)


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
    tool_result: Any
    base_ingredients: List[str]
    recipe: Optional[Dict[str, Any]]
    compliance_attempts: int
    compliance_issues: List[str]
    error: Optional[str]
    output: str


def _fail(message: str) -> Dict[str, Any]:
    """Helper that produces the error state update (``Error: [clear explanation]``)."""
    message = message.strip()
    if not message.startswith(ERROR_PREFIX):
        message = ERROR_PREFIX + message
    return {"error": message, "output": message}


def _describe_exception(exc: Exception, context: str) -> str:
    if isinstance(exc, (httpx.TimeoutException, TimeoutError)):
        return f"{context} timed out. Please try again later."
    if isinstance(exc, httpx.HTTPStatusError):
        return f"{context} failed with HTTP status {exc.response.status_code}."
    text = str(exc).strip() or exc.__class__.__name__
    if "api key" in text.lower() or "api_key" in text.lower() or "missing" in text.lower():
        return f"{context} failed due to a configuration problem ({text})."
    return f"{context} failed: {text}"


# --------------------------------------------------------------------------- #
# Nodes
# --------------------------------------------------------------------------- #


def validate_query(state: AgentState) -> Dict[str, Any]:
    """BR-01: reject empty, offensive or non-food-related queries."""
    query = (state.get("query") or "").strip()
    if not query:
        return {"is_valid": False, **_fail("the query is empty. Please describe the dish, ingredients or preferences you have.")}

    try:
        llm = get_llm().with_structured_output(QueryValidation)
        verdict: QueryValidation = llm.invoke(
            [
                SystemMessage(
                    content=CHEF_PERSONA
                    + "\n\nClassify the user's query. Decide whether it is related to food/cooking/"
                    "recipes/ingredients/diets and whether it contains offensive language. "
                    "Be strict: greetings, programming questions, math, politics, etc. are NOT food related."
                ),
                HumanMessage(content=query),
            ]
        )
    except Exception as exc:  # BR-07
        return {"is_valid": False, **_fail(_describe_exception(exc, "query validation"))}

    if verdict.is_offensive:
        return {"is_valid": False, **_fail("the query contains offensive language. Please rephrase it respectfully and I will be glad to help with a recipe.")}
    if not verdict.is_food_related:
        return {"is_valid": False, **_fail(f"the query is not related to food or cooking ({verdict.reason}). I can only generate recipes.")}
    return {"is_valid": True, "error": None}


def analyze_query(state: AgentState) -> Dict[str, Any]:
    """BR-02: extract explicit ingredients, dietary restrictions and preferences."""
    try:
        llm = get_llm().with_structured_output(QueryAnalysis)
        analysis: QueryAnalysis = llm.invoke(
            [
                SystemMessage(
                    content=CHEF_PERSONA
                    + "\n\nExtract from the user's query: (1) ingredients that are EXPLICITLY mentioned "
                    "(do not infer or invent any); (2) dietary restrictions; (3) other preferences. "
                    "Return empty lists when nothing is mentioned."
                ),
                HumanMessage(content=state["query"]),
            ]
        )
    except Exception as exc:  # BR-07
        return _fail(_describe_exception(exc, "query analysis"))

    return {
        "query_ingredients": [i.strip() for i in analysis.ingredients if i.strip()],
        "dietary_restrictions": [r.strip() for r in analysis.dietary_restrictions if r.strip()],
        "preferences": [p.strip() for p in analysis.preferences if p.strip()],
    }


def call_search_recipes(state: AgentState) -> Dict[str, Any]:
    """BR-03: the query mentions ingredients -> search recipes on Tavily."""
    ingredients = state.get("query_ingredients") or []
    try:
        results = search_recipes.invoke({"ingredients": ingredients})
    except Exception as exc:  # BR-07
        return {"tool_called": "search_recipes", **_fail(_describe_exception(exc, "recipe search (search_recipes)"))}
    return {
        "tool_called": "search_recipes",
        "tool_result": results,
        "base_ingredients": list(ingredients),
    }


def call_retrieve_pantry(state: AgentState) -> Dict[str, Any]:
    """BR-04: no ingredients in the query -> read the pantry from Notion."""
    try:
        pantry = retrieve_pantry.invoke({})
    except Exception as exc:  # BR-07
        return {"tool_called": "retrieve_pantry", **_fail(_describe_exception(exc, "pantry retrieval (retrieve_pantry)"))}
    pantry = [str(i).strip() for i in (pantry or []) if str(i).strip()]
    return {
        "tool_called": "retrieve_pantry",
        "tool_result": pantry,
        "base_ingredients": pantry,
    }


def check_ingredients(state: AgentState) -> Dict[str, Any]:
    """Exception rule: no identifiable ingredients in the tool result nor in the query -> error."""
    if state.get("base_ingredients"):
        return {}
    if state.get("tool_called") == "retrieve_pantry":
        return _fail("no ingredients were found in your pantry and the query did not mention any, so I cannot generate a recipe.")
    return _fail("no identifiable ingredients were found in the query or in the search results, so I cannot generate a recipe.")


def _format_tool_context(state: AgentState) -> str:
    if state.get("tool_called") == "search_recipes":
        results = state.get("tool_result") or []
        if not results:
            return "No recipes were returned by the search; base the recipe on the query ingredients only."
        lines = ["Reference recipes found on the web (use them as inspiration):"]
        for i, r in enumerate(results, 1):
            lines.append(f"{i}. {r.get('title', '')} ({r.get('url', '')})\n   {r.get('content', '')}")
        return "\n".join(lines)
    pantry = state.get("tool_result") or []
    return "Ingredients available in the user's pantry:\n- " + "\n- ".join(pantry)


def generate_recipe(state: AgentState) -> Dict[str, Any]:
    """BR-05 / BR-06: generate the recipe from the tool result and the query."""
    attempts = int(state.get("compliance_attempts") or 0) + 1
    restrictions = state.get("dietary_restrictions") or []
    preferences = state.get("preferences") or []
    base_ingredients = state.get("base_ingredients") or []
    previous_issues = state.get("compliance_issues") or []

    system = (
        CHEF_PERSONA
        + "\n\nCreate ONE complete recipe following these rules:\n"
        "1. The MAIN ingredients of the recipe MUST be the base ingredients listed below "
        "(from the query or from the pantry). You MUST use the tool result provided.\n"
        "2. You may add extra ingredients ONLY if strictly necessary (e.g. salt, oil, water); "
        "keep extras minimal.\n"
        "3. The recipe MUST respect ALL dietary restrictions. Never include an ingredient that "
        "violates them, not even as an optional item.\n"
        "4. Follow the user's preferences whenever possible.\n"
        "5. Provide ingredients with quantities, required equipment, step-by-step instructions, "
        "nutritional information (calories, protein, carbohydrates, fat, health benefits) and "
        "storage recommendations.\n"
        "6. Write the recipe in the same language as the user's query.\n"
        "7. Never use offensive language."
    )
    if previous_issues:
        system += (
            "\n\nYour previous attempt violated the dietary restrictions:\n- "
            + "\n- ".join(previous_issues)
            + "\nFix every issue in this new version."
        )

    human = (
        f"User query: {state['query']}\n\n"
        f"Base ingredients: {', '.join(base_ingredients)}\n"
        f"Dietary restrictions: {', '.join(restrictions) if restrictions else 'none'}\n"
        f"Preferences: {', '.join(preferences) if preferences else 'none'}\n\n"
        f"Tool used: {state.get('tool_called')}\n"
        f"{_format_tool_context(state)}"
    )

    try:
        llm = get_llm().with_structured_output(Recipe)
        recipe: Recipe = llm.invoke([SystemMessage(content=system), HumanMessage(content=human)])
    except Exception as exc:  # BR-07
        return _fail(_describe_exception(exc, "recipe generation"))

    return {"recipe": recipe.model_dump(), "compliance_attempts": attempts}


def verify_restrictions(state: AgentState) -> Dict[str, Any]:
    """Constraint: never deliver a recipe that violates the dietary restrictions."""
    restrictions = state.get("dietary_restrictions") or []
    recipe = state.get("recipe") or {}
    if not restrictions:
        return {"compliance_issues": []}

    try:
        llm = get_llm().with_structured_output(ComplianceCheck)
        check: ComplianceCheck = llm.invoke(
            [
                SystemMessage(
                    content="You are a strict dietary compliance auditor. Given a recipe and a list of "
                    "dietary restrictions, report whether EVERY ingredient and step respects ALL "
                    "restrictions. List concrete violations if any."
                ),
                HumanMessage(
                    content=f"Dietary restrictions: {', '.join(restrictions)}\n\n"
                    f"Recipe title: {recipe.get('title')}\n"
                    f"Ingredients:\n- " + "\n- ".join(recipe.get("ingredients", []))
                    + "\n\nInstructions:\n- " + "\n- ".join(recipe.get("instructions", []))
                ),
            ]
        )
    except Exception as exc:  # BR-07
        return _fail(_describe_exception(exc, "dietary compliance verification"))

    if check.compliant:
        return {"compliance_issues": []}

    if int(state.get("compliance_attempts") or 0) >= MAX_COMPLIANCE_ATTEMPTS:
        return _fail(
            "I could not generate a recipe that respects your dietary restrictions ("
            + "; ".join(check.issues) + ")."
        )
    return {"compliance_issues": list(check.issues)}


def format_output(state: AgentState) -> Dict[str, Any]:
    """Renders the recipe in the format required by the specification."""
    return {"output": format_recipe(state["recipe"]), "error": None}


def format_recipe(recipe: Dict[str, Any]) -> str:
    nutrition = recipe.get("nutrition") or {}
    benefits = nutrition.get("health_benefits") or []
    lines: List[str] = []
    lines.append(f"Título: {recipe.get('title', '')}")
    if recipe.get("servings"):
        lines.append(f"Porções: {recipe['servings']}")
    lines.append("")
    lines.append("Ingredientes:")
    lines.extend(f"- {item}" for item in recipe.get("ingredients", []))
    lines.append("")
    lines.append("Equipamentos:")
    lines.extend(f"- {item}" for item in recipe.get("equipment", []))
    lines.append("")
    lines.append("Instruções:")
    lines.extend(f"{i}. {step}" for i, step in enumerate(recipe.get("instructions", []), 1))
    lines.append("")
    lines.append("Nutrição:")
    lines.append(f"- Calorias: {nutrition.get('calories', '')}")
    lines.append(f"- Proteínas: {nutrition.get('protein', '')}")
    lines.append(f"- Carboidratos: {nutrition.get('carbohydrates', '')}")
    lines.append(f"- Gorduras: {nutrition.get('fat', '')}")
    if benefits:
        lines.append("- Benefícios à saúde: " + "; ".join(benefits))
    lines.append("")
    lines.append(f"Armazenamento: {recipe.get('storage', '')}")
    return "\n".join(lines)


# --------------------------------------------------------------------------- #
# Routing
# --------------------------------------------------------------------------- #


def _has_error(state: AgentState) -> bool:
    return bool(state.get("error"))


def route_after_validation(state: AgentState) -> str:
    return "error" if _has_error(state) or not state.get("is_valid") else "analyze_query"


def route_tool(state: AgentState) -> str:
    """BR-02/03/04: exactly ONE tool is chosen, based on the presence of ingredients."""
    if _has_error(state):
        return "error"
    return "call_search_recipes" if state.get("query_ingredients") else "call_retrieve_pantry"


def route_after_tool(state: AgentState) -> str:
    return "error" if _has_error(state) else "check_ingredients"


def route_after_check(state: AgentState) -> str:
    return "error" if _has_error(state) else "generate_recipe"


def route_after_generate(state: AgentState) -> str:
    return "error" if _has_error(state) else "verify_restrictions"


def route_after_verify(state: AgentState) -> str:
    if _has_error(state):
        return "error"
    if state.get("compliance_issues"):
        return "generate_recipe"
    return "format_output"


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
    workflow.add_node("check_ingredients", check_ingredients)
    workflow.add_node("generate_recipe", generate_recipe)
    workflow.add_node("verify_restrictions", verify_restrictions)
    workflow.add_node("format_output", format_output)

    workflow.set_entry_point("validate_query")
    workflow.add_conditional_edges(
        "validate_query", route_after_validation, {"analyze_query": "analyze_query", "error": END}
    )
    workflow.add_conditional_edges(
        "analyze_query",
        route_tool,
        {"call_search_recipes": "call_search_recipes", "call_retrieve_pantry": "call_retrieve_pantry", "error": END},
    )
    workflow.add_conditional_edges(
        "call_search_recipes", route_after_tool, {"check_ingredients": "check_ingredients", "error": END}
    )
    workflow.add_conditional_edges(
        "call_retrieve_pantry", route_after_tool, {"check_ingredients": "check_ingredients", "error": END}
    )
    workflow.add_conditional_edges(
        "check_ingredients", route_after_check, {"generate_recipe": "generate_recipe", "error": END}
    )
    workflow.add_conditional_edges(
        "generate_recipe", route_after_generate, {"verify_restrictions": "verify_restrictions", "error": END}
    )
    workflow.add_conditional_edges(
        "verify_restrictions",
        route_after_verify,
        {"generate_recipe": "generate_recipe", "format_output": "format_output", "error": END},
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
        tool_result=None,
        base_ingredients=[],
        recipe=None,
        compliance_attempts=0,
        compliance_issues=[],
        error=None,
        output="",
    )


def agent(query: str) -> dict:
    """
    Entrypoint. Runs the recipe-generation workflow for ``query`` and returns
    the final state as a ``dict``.

    ``state["output"]`` holds either the formatted recipe or a string starting
    with ``"Error: "``. ``state["error"]`` is ``None`` on success.
    """
    state = _initial_state(query)

    if not os.getenv("OPENAI_API_KEY", "").strip():
        state.update(_fail("missing OPENAI_API_KEY; the language model cannot be reached."))
        return dict(state)

    try:
        graph = build_graph()
        final_state = graph.invoke(state)
    except Exception as exc:  # BR-07 - last line of defence
        state.update(_fail(_describe_exception(exc, "agent execution")))
        return dict(state)

    result = dict(final_state)
    if not result.get("output"):
        result.update(_fail("the agent finished without producing a recipe."))
    return result


if __name__ == "__main__":
    import sys

    user_query = " ".join(sys.argv[1:]) or "Quero um jantar vegano com grão-de-bico e espinafre"
    print(agent(user_query)["output"])

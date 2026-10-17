import os
import json
from typing import TypedDict, Annotated, Any

from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage, ToolMessage
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode

import requests


# ── Tools ──

@tool
def retrieve_pantry() -> str:
    """Obtain the ingredients available in the user's pantry from a Notion page."""
    try:
        notion_api_key = os.environ.get("NOTION_API_KEY")
        page_id = os.environ.get("NOTION_PAGE_ID")
        if not notion_api_key or not page_id:
            return "Error: Missing Notion credentials (NOTION_API_KEY or NOTION_PAGE_ID)."

        url = f"https://api.notion.com/v1/blocks/{page_id}/children?page_size=100"
        headers = {
            "Authorization": f"Bearer {notion_api_key}",
            "Notion-Version": "2022-06-28",
        }
        resp = requests.get(url, headers=headers, timeout=30)
        if resp.status_code != 200:
            return f"Error: Notion API returned status {resp.status_code}."

        data = resp.json()
        ingredients = []
        for block in data.get("results", []):
            btype = block.get("type", "")
            if btype in ("paragraph", "bulleted_list_item", "numbered_list_item"):
                texts = block.get(btype, {}).get("rich_text", [])
                for t in texts:
                    plain = t.get("plain_text", "").strip()
                    if plain:
                        ingredients.append(plain)
        return json.dumps(ingredients)
    except requests.exceptions.Timeout:
        return "Error: Notion request timed out."
    except Exception as e:
        return f"Error: {e}"


@tool
def search_recipes(ingredients: str) -> str:
    """Search for recipes on the web using Tavily, based on the given ingredients.

    Args:
        ingredients: Comma-separated list of ingredients to search recipes for.
    """
    try:
        api_key = os.environ.get("TAVILY_API_KEY")
        if not api_key:
            return "erro: Missing Tavily API key."

        from tavily import TavilyClient
        client = TavilyClient(api_key=api_key)
        query = f"recipe with {ingredients}"
        response = client.search(query=query, max_results=5)
        results = []
        for r in response.get("results", []):
            results.append({
                "title": r.get("title", ""),
                "url": r.get("url", ""),
                "content": r.get("content", "")[:500],
            })
        return json.dumps(results)
    except Exception as e:
        return f"erro: {e}"


# ── State ──

class AgentState(TypedDict):
    query: str
    is_valid: bool
    error_message: str
    identified_ingredients: list[str]
    reference_recipes: list[dict]
    pantry_ingredients: list[str]
    recipe: dict
    messages: list[Any]
    tool_calls_made: list[str]


SYSTEM_PROMPT = """You are a professional chef AI assistant that creates personalized recipes.
You must respect all dietary restrictions and preferences.
A dietary restriction implies ALL derived ingredients — for example, "vegan" excludes all animal products (meat, dairy, eggs, honey, etc.), "gluten-free" excludes wheat, barley, rye, and all products derived from them.
An ingredient mentioned under negation, absence, or hypothesis (e.g., "I don't have", "without", "no", "if I had") is NOT an available ingredient.
Always respond in the structured JSON format requested."""


tools = [retrieve_pantry, search_recipes]


def _get_llm():
    return ChatOpenAI(model="gpt-4o", temperature=0)


# ── Node functions ──

def validate_query(state: AgentState) -> AgentState:
    """B1: Validate the user query."""
    query = state.get("query", "").strip()
    if not query:
        return {**state, "is_valid": False, "error_message": "Error: The query is empty."}

    llm = _get_llm()
    resp = llm.invoke([
        SystemMessage(content=SYSTEM_PROMPT),
        HumanMessage(content=f"""Classify the following query as "valid" or "invalid".
A query is invalid if it is:
- Empty or nonsensical
- Offensive or inappropriate
- Completely unrelated to food, cooking, or recipes

Query: "{query}"

Respond with exactly one JSON object: {{"valid": true}} or {{"valid": false, "reason": "..."}}"""),
    ])
    try:
        result = json.loads(resp.content.strip().removeprefix("```json").removesuffix("```").strip())
    except json.JSONDecodeError:
        content = resp.content.lower()
        if "true" in content:
            result = {"valid": True}
        else:
            result = {"valid": False, "reason": "Could not determine query validity."}

    if not result.get("valid", False):
        reason = result.get("reason", "The query is not related to food or cooking.")
        return {**state, "is_valid": False, "error_message": f"Error: {reason}"}
    return {**state, "is_valid": True, "error_message": ""}


def identify_ingredients(state: AgentState) -> AgentState:
    """B2: Identify compatible ingredients from the query."""
    llm = _get_llm()
    resp = llm.invoke([
        SystemMessage(content=SYSTEM_PROMPT),
        HumanMessage(content=f"""From the following query, extract:
1. All ingredients that the user explicitly mentions as available.
2. Any dietary restrictions or preferences.

Then filter out any ingredient that violates the dietary restrictions (including all derived ingredients).
Also exclude any ingredient mentioned under negation, absence, or hypothesis (e.g., "I don't have X", "without X", "no X", "if I had X").

Query: "{state['query']}"

Respond with exactly one JSON object:
{{"ingredients": ["ingredient1", "ingredient2", ...], "restrictions": ["restriction1", ...], "preferences": "..."}}
If no compatible ingredients are found, return {{"ingredients": [], "restrictions": [...], "preferences": "..."}}"""),
    ])
    try:
        result = json.loads(resp.content.strip().removeprefix("```json").removesuffix("```").strip())
    except json.JSONDecodeError:
        result = {"ingredients": [], "restrictions": [], "preferences": ""}

    return {
        **state,
        "identified_ingredients": result.get("ingredients", []),
        "messages": state.get("messages", []) + [
            {"role": "system", "restrictions": result.get("restrictions", []),
             "preferences": result.get("preferences", "")}
        ],
    }


def search_recipes_node(state: AgentState) -> AgentState:
    """B3: Search reference recipes using the search_recipes tool."""
    ingredients_str = ", ".join(state["identified_ingredients"])

    llm = _get_llm().bind_tools([search_recipes])
    resp = llm.invoke([
        SystemMessage(content="You are a chef assistant. Call the search_recipes tool with the given ingredients."),
        HumanMessage(content=f"Search for recipes using these ingredients: {ingredients_str}"),
    ])

    tool_calls_made = state.get("tool_calls_made", []) + ["search_recipes"]

    if resp.tool_calls:
        tool_node = ToolNode([search_recipes])
        tool_msg = tool_node.invoke({"messages": [resp]})
        last_msg = tool_msg["messages"][-1]
        content = last_msg.content if hasattr(last_msg, "content") else str(last_msg)

        if isinstance(content, str) and content.startswith("erro:"):
            return {
                **state,
                "error_message": f"Error: {content[5:].strip()}",
                "reference_recipes": [],
                "tool_calls_made": tool_calls_made,
            }

        try:
            recipes = json.loads(content) if isinstance(content, str) else content
        except (json.JSONDecodeError, TypeError):
            recipes = []

        return {**state, "reference_recipes": recipes if isinstance(recipes, list) else [],
                "tool_calls_made": tool_calls_made}
    else:
        result = search_recipes.invoke({"ingredients": ingredients_str})
        tool_calls_made = state.get("tool_calls_made", []) + ["search_recipes"]
        if isinstance(result, str) and result.startswith("erro:"):
            return {**state, "error_message": f"Error: {result[5:].strip()}",
                    "reference_recipes": [], "tool_calls_made": tool_calls_made}
        try:
            recipes = json.loads(result) if isinstance(result, str) else result
        except (json.JSONDecodeError, TypeError):
            recipes = []
        return {**state, "reference_recipes": recipes if isinstance(recipes, list) else [],
                "tool_calls_made": tool_calls_made}


def retrieve_pantry_node(state: AgentState) -> AgentState:
    """B4: Retrieve pantry ingredients and filter by dietary restrictions."""
    llm = _get_llm().bind_tools([retrieve_pantry])
    resp = llm.invoke([
        SystemMessage(content="You are a chef assistant. Call the retrieve_pantry tool to get the user's pantry."),
        HumanMessage(content="Retrieve the pantry ingredients."),
    ])

    tool_calls_made = state.get("tool_calls_made", []) + ["retrieve_pantry"]

    if resp.tool_calls:
        tool_node = ToolNode([retrieve_pantry])
        tool_msg = tool_node.invoke({"messages": [resp]})
        last_msg = tool_msg["messages"][-1]
        content = last_msg.content if hasattr(last_msg, "content") else str(last_msg)
    else:
        content = retrieve_pantry.invoke({})

    if isinstance(content, str) and content.startswith("Error:"):
        return {**state, "error_message": content, "pantry_ingredients": [],
                "tool_calls_made": tool_calls_made}

    try:
        pantry = json.loads(content) if isinstance(content, str) else content
    except (json.JSONDecodeError, TypeError):
        pantry = []

    if not isinstance(pantry, list):
        pantry = []

    restrictions_info = ""
    for m in state.get("messages", []):
        if isinstance(m, dict) and m.get("role") == "system" and "restrictions" in m:
            restrictions_info = json.dumps(m["restrictions"])

    if pantry and restrictions_info and restrictions_info != "[]":
        filter_llm = _get_llm()
        filter_resp = filter_llm.invoke([
            SystemMessage(content=SYSTEM_PROMPT),
            HumanMessage(content=f"""Filter the following pantry ingredients to keep only those compatible with these dietary restrictions: {restrictions_info}

Remember: a dietary restriction implies ALL derived ingredients.

Pantry ingredients: {json.dumps(pantry)}

Respond with exactly one JSON array of compatible ingredients: ["ingredient1", ...]
If none are compatible, respond with: []"""),
        ])
        try:
            filtered = json.loads(filter_resp.content.strip().removeprefix("```json").removesuffix("```").strip())
            pantry = filtered if isinstance(filtered, list) else []
        except json.JSONDecodeError:
            pass

    return {**state, "pantry_ingredients": pantry, "tool_calls_made": tool_calls_made}


def check_no_ingredients(state: AgentState) -> AgentState:
    """B5: Check if pantry returned no compatible ingredients."""
    return {**state, "error_message": "Error: No compatible ingredients found in pantry for the given dietary restrictions."}


def generate_recipe(state: AgentState) -> AgentState:
    """B7: Generate the complete recipe."""
    ingredients = state.get("identified_ingredients", []) or state.get("pantry_ingredients", [])
    references = state.get("reference_recipes", [])

    ref_text = ""
    if references:
        ref_text = "\n\nReference recipes for inspiration (do NOT copy them, use as guidance only):\n"
        for r in references[:5]:
            ref_text += f"- {r.get('title', '')}: {r.get('content', '')[:200]}\n"

    restrictions_info = ""
    preferences_info = ""
    for m in state.get("messages", []):
        if isinstance(m, dict) and m.get("role") == "system":
            if m.get("restrictions"):
                restrictions_info = f"\nDietary restrictions: {json.dumps(m['restrictions'])}"
            if m.get("preferences"):
                preferences_info = f"\nPreferences: {m['preferences']}"

    llm = _get_llm()
    resp = llm.invoke([
        SystemMessage(content=SYSTEM_PROMPT),
        HumanMessage(content=f"""Create a complete recipe using these available ingredients: {json.dumps(ingredients)}
{restrictions_info}{preferences_info}{ref_text}

User query: "{state['query']}"

You MUST respond with exactly one JSON object with these fields:
{{
  "title": "Recipe Title",
  "ingredients": ["quantity ingredient1", "quantity ingredient2", ...],
  "equipment": ["equipment1", "equipment2", ...],
  "instructions": ["Step 1: ...", "Step 2: ...", ...],
  "storage": "Storage recommendations..."
}}

Rules:
- All fields must be present and non-empty.
- No ingredient may violate the dietary restrictions.
- Use at least one of the provided ingredients.
- Be creative and practical."""),
    ])

    try:
        recipe = json.loads(resp.content.strip().removeprefix("```json").removesuffix("```").strip())
    except json.JSONDecodeError:
        recipe = {
            "title": "Recipe",
            "ingredients": ingredients,
            "equipment": ["Basic kitchen equipment"],
            "instructions": [resp.content],
            "storage": "Store in an airtight container in the refrigerator.",
        }

    return {**state, "recipe": recipe}


# ── Routing ──

def route_after_validation(state: AgentState) -> str:
    if not state.get("is_valid", False):
        return "end"
    return "identify_ingredients"


def route_after_ingredients(state: AgentState) -> str:
    if state.get("identified_ingredients"):
        return "search_recipes_node"
    return "retrieve_pantry_node"


def route_after_search(state: AgentState) -> str:
    if state.get("error_message"):
        return "end"
    return "generate_recipe"


def route_after_pantry(state: AgentState) -> str:
    if state.get("error_message"):
        return "end"
    if not state.get("pantry_ingredients"):
        return "check_no_ingredients"
    return "generate_recipe"


# ── Graph ──

def _build_graph():
    graph = StateGraph(AgentState)

    graph.add_node("validate_query", validate_query)
    graph.add_node("identify_ingredients", identify_ingredients)
    graph.add_node("search_recipes_node", search_recipes_node)
    graph.add_node("retrieve_pantry_node", retrieve_pantry_node)
    graph.add_node("check_no_ingredients", check_no_ingredients)
    graph.add_node("generate_recipe", generate_recipe)

    graph.set_entry_point("validate_query")

    graph.add_conditional_edges("validate_query", route_after_validation,
                                {"end": END, "identify_ingredients": "identify_ingredients"})
    graph.add_conditional_edges("identify_ingredients", route_after_ingredients,
                                {"search_recipes_node": "search_recipes_node",
                                 "retrieve_pantry_node": "retrieve_pantry_node"})
    graph.add_conditional_edges("search_recipes_node", route_after_search,
                                {"end": END, "generate_recipe": "generate_recipe"})
    graph.add_conditional_edges("retrieve_pantry_node", route_after_pantry,
                                {"end": END, "check_no_ingredients": "check_no_ingredients",
                                 "generate_recipe": "generate_recipe"})
    graph.add_edge("check_no_ingredients", END)
    graph.add_edge("generate_recipe", END)

    return graph.compile()


_compiled_graph = None


def agent(query: str) -> dict:
    """Entrypoint for the recipe agent."""
    global _compiled_graph
    if _compiled_graph is None:
        _compiled_graph = _build_graph()

    initial_state: AgentState = {
        "query": query,
        "is_valid": False,
        "error_message": "",
        "identified_ingredients": [],
        "reference_recipes": [],
        "pantry_ingredients": [],
        "recipe": {},
        "messages": [],
        "tool_calls_made": [],
    }

    try:
        result = _compiled_graph.invoke(initial_state)
        return dict(result)
    except Exception as e:
        return {
            **initial_state,
            "error_message": f"Error: {e}",
        }

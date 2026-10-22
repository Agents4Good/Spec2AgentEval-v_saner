"""Personalized recipe generator agent.

Persona: a chef that creates personalized recipes adapted to the ingredients
provided in the user's query (or in their pantry) while respecting the user's
dietary preferences and restrictions. It returns the complete recipe with
ingredients, equipment, instructions and storage recommendations.

Framework: LangGraph
Model: GPT-4o via ChatOpenAI
Tools: LangChain tools, executed exclusively through their interface.
Entrypoint: agent(query: str) -> response: str
"""

import json
import os
from typing import List, TypedDict

import requests
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langgraph.graph import END, StateGraph

# ---------------------------------------------------------------------------
# Tools
# ---------------------------------------------------------------------------


@tool
def retrieve_pantry() -> list:
    """Obtain the ingredients available in the user's pantry from a Notion page.

    Takes no input. Returns the complete list of pantry ingredients; an empty
    pantry returns an empty list. On missing credentials, an inaccessible page
    or a timeout, returns a string beginning with ``Error:`` explaining the
    failure.
    """
    notion_api_key = os.environ.get("NOTION_API_KEY")
    notion_page_id = os.environ.get("NOTION_PAGE_ID")

    if not notion_api_key or not notion_page_id:
        return "Error: missing Notion credentials (NOTION_API_KEY or NOTION_PAGE_ID)."

    url = f"https://api.notion.com/v1/blocks/{notion_page_id}/children"
    headers = {
        "Authorization": f"Bearer {notion_api_key}",
        "Notion-Version": "2022-06-28",
    }

    try:
        resp = requests.get(url, headers=headers, timeout=30)
    except requests.exceptions.Timeout:
        return "Error: timeout while accessing the Notion pantry page."
    except requests.exceptions.RequestException as exc:
        return f"Error: could not access the Notion pantry page ({exc})."

    if resp.status_code != 200:
        return (
            f"Error: Notion pantry page is inaccessible "
            f"(status {resp.status_code})."
        )

    try:
        data = resp.json()
    except ValueError:
        return "Error: invalid response received from the Notion pantry page."

    ingredients: List[str] = []
    for block in data.get("results", []):
        block_type = block.get("type")
        content = block.get(block_type, {}) if block_type else {}
        rich_text = content.get("rich_text", []) if isinstance(content, dict) else []
        text = "".join(rt.get("plain_text", "") for rt in rich_text).strip()
        if text:
            ingredients.append(text)

    return ingredients


@tool
def search_recipes(ingredients: List[str]) -> list:
    """Obtain recipes from the web (via Tavily) as a preparation reference.

    The recipes found are neither a source of base ingredients nor the final
    recipe. Input: the usable ingredients identified from the pantry and the
    query. Returns up to 5 results, each with title, URL and content excerpt;
    no results returns an empty list. On missing credentials, an inaccessible
    page or a timeout, returns a string beginning with ``Error:`` explaining
    the failure.
    """
    tavily_api_key = os.environ.get("TAVILY_API_KEY")
    if not tavily_api_key:
        return "Error: missing Tavily credentials (TAVILY_API_KEY)."

    if not ingredients:
        return []

    query = "recipe with " + ", ".join(ingredients)

    try:
        resp = requests.post(
            "https://api.tavily.com/search",
            json={
                "api_key": tavily_api_key,
                "query": query,
                "max_results": 5,
            },
            timeout=30,
        )
    except requests.exceptions.Timeout:
        return "Error: timeout while searching recipes on Tavily."
    except requests.exceptions.RequestException as exc:
        return f"Error: could not reach the Tavily search service ({exc})."

    if resp.status_code != 200:
        return f"Error: Tavily search service is inaccessible (status {resp.status_code})."

    try:
        data = resp.json()
    except ValueError:
        return "Error: invalid response received from the Tavily search service."

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


TOOLS = {"retrieve_pantry": retrieve_pantry, "search_recipes": search_recipes}


# ---------------------------------------------------------------------------
# LLM
# ---------------------------------------------------------------------------


def _llm() -> ChatOpenAI:
    return ChatOpenAI(model="gpt-4o", temperature=0)


# ---------------------------------------------------------------------------
# Graph state
# ---------------------------------------------------------------------------


class AgentState(TypedDict, total=False):
    query: str
    valid: bool
    pantry: List[str]
    ingredients: List[str]
    search_results: list
    response: str
    done: bool


# ---------------------------------------------------------------------------
# Behaviors (graph nodes)
# ---------------------------------------------------------------------------


def _b1_validate(state: AgentState) -> AgentState:
    """B1 - classify the query as valid or invalid."""
    query = (state.get("query") or "").strip()

    if not query:
        return {"valid": False, "done": True, "response": "Error: the query is empty."}

    try:
        llm = _llm()
        prompt = (
            "You are a strict classifier for a cooking recipe assistant. "
            "Classify the user query as VALID or INVALID.\n"
            "INVALID means: empty, offensive, or unrelated to food/cooking.\n"
            "VALID means: any query that meets none of the invalidity criteria.\n"
            "Respond ONLY with a JSON object: "
            '{"valid": true|false, "reason": "short explanation"}.'
        )
        msg = llm.invoke([SystemMessage(content=prompt), HumanMessage(content=query)])
        parsed = _extract_json(msg.content)
        valid = bool(parsed.get("valid"))
        reason = parsed.get("reason", "the query is not related to food or cooking.")
    except Exception as exc:  # B6 - model failure
        return {
            "valid": False,
            "done": True,
            "response": f"Error: execution failure during query validation ({exc}).",
        }

    if not valid:
        return {
            "valid": False,
            "done": True,
            "response": f"Error: invalid query - {reason}",
        }
    return {"valid": True}


def _b2_pantry(state: AgentState) -> AgentState:
    """B2 - retrieve the ingredients available in the user's pantry.

    ``retrieve_pantry`` is the first tool called and is called exactly once.
    """
    try:
        result = retrieve_pantry.invoke({})
    except Exception as exc:  # B6
        return {
            "done": True,
            "response": f"Error: execution failure during pantry lookup ({exc}).",
        }

    # Tool-level failure signalled with the 'Error:'/'erro:' prefix.
    if isinstance(result, str):
        if result.strip().lower().startswith(("error:", "erro:")):
            return {"done": True, "response": f"Error: {result.split(':', 1)[1].strip()}"}
        return {"done": True, "response": "Error: unexpected pantry response."}

    pantry = result if isinstance(result, list) else []
    pantry = [str(i).strip() for i in pantry if str(i).strip()]
    return {"pantry": pantry}


def _b3_identify_ingredients(state: AgentState) -> AgentState:
    """B3 - identify usable ingredients (query + pantry) compatible with
    the declared dietary restrictions."""
    query = state["query"]
    pantry = state.get("pantry", [])
    try:
        llm = _llm()
        prompt = (
            "You identify the ingredients usable in a recipe for a cooking "
            "assistant. The usable ingredients are those explicitly available "
            "in the user's query PLUS all those returned by the pantry.\n"
            "Rules (invariants):\n"
            "- A dietary restriction implies ALL of its derived ingredients, even "
            "those not mentioned (e.g. vegan excludes meat, dairy, eggs, honey; "
            "gluten-free excludes wheat, barley, rye). Keep ONLY ingredients "
            "compatible with the declared dietary restrictions.\n"
            "- An ingredient mentioned under negation, absence, or hypothesis "
            "(e.g. 'without onion', 'I have no milk', 'if I had eggs') is NOT an "
            "available ingredient and must be excluded.\n"
            "- Normalize regional variants and inflected forms to a canonical "
            "ingredient name.\n"
            'Respond ONLY with a JSON object: {"ingredients": ["..."]}. '
            "If none are identified or compatible, return an empty list."
        )
        msg = llm.invoke(
            [
                SystemMessage(content=prompt),
                HumanMessage(
                    content=(
                        f"Query: {query}\n"
                        f"Pantry ingredients: {json.dumps(pantry, ensure_ascii=False)}"
                    )
                ),
            ]
        )
        parsed = _extract_json(msg.content)
        ingredients = parsed.get("ingredients", [])
        if not isinstance(ingredients, list):
            ingredients = []
        ingredients = [str(i).strip() for i in ingredients if str(i).strip()]
    except Exception as exc:  # B6
        return {
            "done": True,
            "response": f"Error: execution failure during ingredient identification ({exc}).",
        }
    return {"ingredients": ingredients}


def _b4_search(state: AgentState) -> AgentState:
    """B4 - search reference recipes via the search_recipes tool.

    Called exactly once, only after ``retrieve_pantry`` has completed, with the
    list of ingredients from B3.
    """
    try:
        result = search_recipes.invoke({"ingredients": state["ingredients"]})
    except Exception as exc:  # B6
        return {
            "done": True,
            "response": f"Error: execution failure during recipe search ({exc}).",
        }

    # Tool-level failure signalled with the 'Error:'/'erro:' prefix.
    if isinstance(result, str):
        if result.strip().lower().startswith(("error:", "erro:")):
            return {"done": True, "response": f"Error: {result.split(':', 1)[1].strip()}"}
        return {"done": True, "response": "Error: unexpected recipe search response."}

    if not isinstance(result, list):
        result = []
    return {"search_results": result}


def _b5_lack(state: AgentState) -> AgentState:
    """B5 - stop when no compatible ingredient exists in query or pantry."""
    return {
        "done": True,
        "response": (
            "Error: there are no ingredients compatible with the declared "
            "dietary restrictions, either in your query or in your pantry."
        ),
    }


def _b7_generate(state: AgentState) -> AgentState:
    """B7 - generate the complete recipe."""
    try:
        llm = _llm()
        references = state.get("search_results") or []
        ref_text = (
            json.dumps(references, ensure_ascii=False)
            if references
            else "none available"
        )
        prompt = (
            "You are a professional chef. Create a complete, personalized recipe.\n"
            "You MUST use the compatible ingredients and respect the user's "
            "preferences and dietary restrictions. A dietary restriction implies "
            "all of its derived ingredients - never include any of them.\n"
            "The recipe must include at least one of the provided compatible "
            "ingredients. Use the reference recipes only as a preparation guide, "
            "not as a source of base ingredients.\n"
            "Return the recipe with ALL of these non-empty fields, each clearly "
            "labelled:\n"
            "Title: <recipe title>\n"
            "Ingredients: <list of ingredients with quantities>\n"
            "Equipment: <list of required equipment>\n"
            "Instructions: <step-by-step preparation instructions>\n"
            "Storage: <storage recommendations>"
        )
        msg = llm.invoke(
            [
                SystemMessage(content=prompt),
                HumanMessage(
                    content=(
                        f"User query: {state['query']}\n"
                        f"Compatible ingredients: {json.dumps(state.get('ingredients', []), ensure_ascii=False)}\n"
                        f"Reference recipes: {ref_text}"
                    )
                ),
            ]
        )
        recipe = msg.content
        if isinstance(recipe, list):
            recipe = "".join(
                part.get("text", "") if isinstance(part, dict) else str(part)
                for part in recipe
            )
    except Exception as exc:  # B6
        return {
            "done": True,
            "response": f"Error: execution failure during recipe generation ({exc}).",
        }
    return {"done": True, "response": str(recipe)}


# ---------------------------------------------------------------------------
# Routing
# ---------------------------------------------------------------------------


def _route_after_b1(state: AgentState) -> str:
    return "b2" if state.get("valid") else END


def _route_after_b2(state: AgentState) -> str:
    return END if state.get("done") else "b3"


def _route_after_b3(state: AgentState) -> str:
    if state.get("done"):
        return END
    return "b4" if state.get("ingredients") else "b5"


def _route_after_b4(state: AgentState) -> str:
    return END if state.get("done") else "b7"


def _build_graph():
    graph = StateGraph(AgentState)
    graph.add_node("b1", _b1_validate)
    graph.add_node("b2", _b2_pantry)
    graph.add_node("b3", _b3_identify_ingredients)
    graph.add_node("b4", _b4_search)
    graph.add_node("b5", _b5_lack)
    graph.add_node("b7", _b7_generate)

    graph.set_entry_point("b1")
    graph.add_conditional_edges("b1", _route_after_b1, {"b2": "b2", END: END})
    graph.add_conditional_edges("b2", _route_after_b2, {"b3": "b3", END: END})
    graph.add_conditional_edges("b3", _route_after_b3, {"b4": "b4", "b5": "b5", END: END})
    graph.add_conditional_edges("b4", _route_after_b4, {"b7": "b7", END: END})
    graph.add_edge("b5", END)
    graph.add_edge("b7", END)
    return graph.compile()


_GRAPH = None


def _get_graph():
    global _GRAPH
    if _GRAPH is None:
        _GRAPH = _build_graph()
    return _GRAPH


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _extract_json(content) -> dict:
    """Best-effort extraction of a JSON object from an LLM response."""
    if isinstance(content, list):
        content = "".join(
            part.get("text", "") if isinstance(part, dict) else str(part)
            for part in content
        )
    text = str(content).strip()
    if text.startswith("```"):
        text = text.strip("`")
        if text.lstrip().lower().startswith("json"):
            text = text.lstrip()[4:]
    try:
        return json.loads(text)
    except (ValueError, TypeError):
        start = text.find("{")
        end = text.rfind("}")
        if start != -1 and end != -1 and end > start:
            try:
                return json.loads(text[start : end + 1])
            except (ValueError, TypeError):
                pass
    return {}


# ---------------------------------------------------------------------------
# Entrypoint
# ---------------------------------------------------------------------------


def agent(query: str) -> str:
    """Generate a personalized recipe for the given natural-language query."""
    try:
        final_state = _get_graph().invoke({"query": query})
    except Exception as exc:  # B6 - never abort with an unhandled exception
        return f"Error: execution failure ({exc})."
    return final_state.get("response", "Error: no response was produced.")


if __name__ == "__main__":
    import sys

    user_query = " ".join(sys.argv[1:]) or "I have chicken and rice, make me dinner."
    print(agent(user_query))

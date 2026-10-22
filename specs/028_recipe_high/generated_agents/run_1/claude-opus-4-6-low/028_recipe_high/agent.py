import os
import json
from typing import TypedDict, Annotated, Any
from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage, ToolMessage
from langgraph.graph import StateGraph, END


# --- Tools ---

@tool
def retrieve_pantry() -> str:
    """Obtain the ingredients available in the user's pantry from a Notion page.
    Returns a JSON list of ingredient strings, or an error message starting with 'Error:'."""
    try:
        from notion_client import Client

        api_key = os.environ.get("NOTION_API_KEY")
        page_id = os.environ.get("NOTION_PAGE_ID")
        if not api_key:
            return "Error: Missing NOTION_API_KEY credentials."
        if not page_id:
            return "Error: Missing NOTION_PAGE_ID credentials."

        notion = Client(auth=api_key)
        blocks = notion.blocks.children.list(block_id=page_id)
        ingredients = []
        for block in blocks.get("results", []):
            btype = block.get("type", "")
            if btype == "bulleted_list_item":
                texts = block[btype].get("rich_text", [])
                for t in texts:
                    text = t.get("plain_text", "").strip()
                    if text:
                        ingredients.append(text)
            elif btype == "paragraph":
                texts = block[btype].get("rich_text", [])
                for t in texts:
                    text = t.get("plain_text", "").strip()
                    if text:
                        for line in text.split("\n"):
                            line = line.strip().lstrip("•-* ").strip()
                            if line:
                                ingredients.append(line)
        return json.dumps(ingredients)
    except Exception as e:
        return f"Error: {str(e)}"


@tool
def search_recipes(ingredients: str) -> str:
    """Search for recipes on the web using Tavily, based on the given ingredients.
    Input: comma-separated list of ingredients.
    Returns a JSON list of up to 5 results with title, url, and content, or an error message starting with 'Error:'."""
    try:
        from tavily import TavilyClient

        api_key = os.environ.get("TAVILY_API_KEY")
        if not api_key:
            return "Error: Missing TAVILY_API_KEY credentials."

        client = TavilyClient(api_key=api_key)
        query = f"recipe with {ingredients}"
        response = client.search(query=query, max_results=5)
        results = []
        for r in response.get("results", []):
            results.append({
                "title": r.get("title", ""),
                "url": r.get("url", ""),
                "content": r.get("content", "")
            })
        return json.dumps(results)
    except Exception as e:
        return f"Error: {str(e)}"


# --- State ---

class AgentState(TypedDict):
    query: str
    is_valid: bool
    validation_message: str
    pantry_ingredients: list[str]
    compatible_ingredients: list[str]
    reference_recipes: list[dict]
    result: str
    tools_called: list[str]
    error: bool


# --- LLM ---

def get_llm():
    return ChatOpenAI(model="gpt-4o", temperature=0)


# --- Node functions ---

def validate_query(state: AgentState) -> dict:
    query = state["query"].strip()
    if not query:
        return {"is_valid": False, "validation_message": "Error: The query is empty.", "error": True}

    llm = get_llm()
    resp = llm.invoke([
        SystemMessage(content=(
            "You are a query validator for a recipe agent. Classify the query as valid or invalid. "
            "Invalid queries are: empty, offensive/abusive, or completely unrelated to food/cooking/recipes/ingredients/diet. "
            "Respond with exactly 'VALID' or 'INVALID: <reason>'."
        )),
        HumanMessage(content=query)
    ])
    text = resp.content.strip()
    if text.startswith("INVALID"):
        reason = text.split(":", 1)[1].strip() if ":" in text else "The query is not related to food or cooking."
        return {"is_valid": False, "validation_message": f"Error: {reason}", "error": True}
    return {"is_valid": True, "validation_message": "", "error": False}


def lookup_pantry(state: AgentState) -> dict:
    try:
        result = retrieve_pantry.invoke({})
        tools_called = state.get("tools_called", []) + ["retrieve_pantry"]
        if isinstance(result, str) and result.startswith("Error:"):
            return {"pantry_ingredients": [], "tools_called": tools_called,
                    "result": result, "error": True}
        ingredients = json.loads(result) if isinstance(result, str) else result
        if not isinstance(ingredients, list):
            ingredients = []
        return {"pantry_ingredients": ingredients, "tools_called": tools_called, "error": False}
    except Exception as e:
        tools_called = state.get("tools_called", []) + ["retrieve_pantry"]
        return {"pantry_ingredients": [], "tools_called": tools_called,
                "result": f"Error: Failed to retrieve pantry: {str(e)}", "error": True}


def identify_ingredients(state: AgentState) -> dict:
    query = state["query"]
    pantry = state.get("pantry_ingredients", [])

    llm = get_llm()
    resp = llm.invoke([
        SystemMessage(content=(
            "You are a culinary ingredient analyst. Given a user's query and pantry ingredients, identify all usable ingredients.\n\n"
            "Rules:\n"
            "1. Include ingredients explicitly mentioned in the query as available.\n"
            "2. Include all pantry ingredients.\n"
            "3. EXCLUDE any ingredient mentioned under negation, absence, or hypothesis (e.g., 'no eggs', 'without milk', 'if I had butter').\n"
            "4. EXCLUDE any ingredient incompatible with declared dietary restrictions. A dietary restriction implies ALL derived ingredients "
            "(e.g., 'vegan' excludes meat, dairy, eggs, honey, gelatin, and all their derivatives; 'gluten-free' excludes wheat, barley, rye, and all their derivatives).\n"
            "5. Consider regional variants and inflected forms as the same ingredient.\n\n"
            "Return a JSON array of compatible ingredient strings. If none are compatible, return an empty array []."
        )),
        HumanMessage(content=f"Query: {query}\nPantry ingredients: {json.dumps(pantry)}")
    ])
    text = resp.content.strip()
    if "```" in text:
        text = text.split("```")[1]
        if text.startswith("json"):
            text = text[4:]
        text = text.strip()
    try:
        ingredients = json.loads(text)
        if not isinstance(ingredients, list):
            ingredients = []
    except json.JSONDecodeError:
        ingredients = []
    return {"compatible_ingredients": ingredients, "error": False}


def search_reference_recipes(state: AgentState) -> dict:
    ingredients = state.get("compatible_ingredients", [])
    try:
        ingredients_str = ", ".join(ingredients)
        result = search_recipes.invoke({"ingredients": ingredients_str})
        tools_called = state.get("tools_called", []) + ["search_recipes"]
        if isinstance(result, str) and result.startswith("Error:"):
            return {"reference_recipes": [], "tools_called": tools_called,
                    "result": result, "error": True}
        recipes = json.loads(result) if isinstance(result, str) else result
        if not isinstance(recipes, list):
            recipes = []
        return {"reference_recipes": recipes, "tools_called": tools_called, "error": False}
    except Exception as e:
        tools_called = state.get("tools_called", []) + ["search_recipes"]
        return {"reference_recipes": [], "tools_called": tools_called,
                "result": f"Error: Failed to search recipes: {str(e)}", "error": True}


def generate_recipe(state: AgentState) -> dict:
    query = state["query"]
    ingredients = state.get("compatible_ingredients", [])
    references = state.get("reference_recipes", [])

    ref_text = ""
    if references:
        for r in references:
            ref_text += f"- {r.get('title', '')}: {r.get('content', '')[:200]}\n"

    llm = get_llm()
    resp = llm.invoke([
        SystemMessage(content=(
            "You are a professional chef creating personalized recipes. Generate a complete recipe using ONLY the provided "
            "compatible ingredients (you may add basic seasonings like salt, pepper, oil, water). Respect any dietary restrictions "
            "mentioned in the query. Use the reference recipes for inspiration but create an original recipe.\n\n"
            "The recipe MUST include ALL of these sections:\n"
            "**Title:** A creative recipe title\n"
            "**Ingredients:** List each ingredient with quantity\n"
            "**Equipment:** List all required equipment\n"
            "**Instructions:** Step-by-step preparation instructions\n"
            "**Storage:** Storage recommendations\n\n"
            "Do NOT include any ingredient that violates the dietary restrictions."
        )),
        HumanMessage(content=(
            f"User request: {query}\n"
            f"Compatible ingredients: {json.dumps(ingredients)}\n"
            f"Reference recipes:\n{ref_text if ref_text else 'None available.'}"
        ))
    ])
    return {"result": resp.content.strip(), "error": False}


def return_error(state: AgentState) -> dict:
    msg = state.get("validation_message", "") or state.get("result", "")
    if not msg.startswith("Error:"):
        msg = f"Error: {msg}" if msg else "Error: An unknown error occurred."
    return {"result": msg}


def no_ingredients_error(state: AgentState) -> dict:
    return {"result": "Error: No compatible ingredients were found. Cannot generate a recipe.", "error": True}


# --- Routing ---

def after_validation(state: AgentState) -> str:
    if not state.get("is_valid", False):
        return "error"
    return "lookup_pantry"


def after_pantry(state: AgentState) -> str:
    if state.get("error", False):
        return "error"
    return "identify_ingredients"


def after_ingredients(state: AgentState) -> str:
    if not state.get("compatible_ingredients"):
        return "no_ingredients"
    return "search_recipes"


def after_search(state: AgentState) -> str:
    if state.get("error", False):
        return "error"
    return "generate_recipe"


# --- Graph ---

def build_graph():
    graph = StateGraph(AgentState)

    graph.add_node("validate_query", validate_query)
    graph.add_node("lookup_pantry", lookup_pantry)
    graph.add_node("identify_ingredients", identify_ingredients)
    graph.add_node("search_recipes", search_reference_recipes)
    graph.add_node("generate_recipe", generate_recipe)
    graph.add_node("error", return_error)
    graph.add_node("no_ingredients", no_ingredients_error)

    graph.set_entry_point("validate_query")

    graph.add_conditional_edges("validate_query", after_validation, {
        "error": "error",
        "lookup_pantry": "lookup_pantry"
    })
    graph.add_conditional_edges("lookup_pantry", after_pantry, {
        "error": "error",
        "identify_ingredients": "identify_ingredients"
    })
    graph.add_conditional_edges("identify_ingredients", after_ingredients, {
        "no_ingredients": "no_ingredients",
        "search_recipes": "search_recipes"
    })
    graph.add_conditional_edges("search_recipes", after_search, {
        "error": "error",
        "generate_recipe": "generate_recipe"
    })

    graph.add_edge("generate_recipe", END)
    graph.add_edge("error", END)
    graph.add_edge("no_ingredients", END)

    return graph.compile()


# --- Entrypoint ---

def agent(query: str) -> str:
    try:
        app = build_graph()
        initial_state: AgentState = {
            "query": query,
            "is_valid": False,
            "validation_message": "",
            "pantry_ingredients": [],
            "compatible_ingredients": [],
            "reference_recipes": [],
            "result": "",
            "tools_called": [],
            "error": False,
        }
        final_state = app.invoke(initial_state)
        return final_state.get("result", "Error: No result produced.")
    except Exception as e:
        return f"Error: {str(e)}"

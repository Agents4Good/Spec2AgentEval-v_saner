import os
import json
from typing import TypedDict, Annotated, Any

from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode


# --------------- Tools ---------------

@tool
def retrieve_pantry() -> str:
    """Obtain the ingredients available in the user's pantry from a Notion page.
    Returns a JSON list of ingredient strings, or an error message starting with 'Error:'."""
    try:
        from notion_client import Client

        api_key = os.environ.get("NOTION_API_KEY")
        page_id = os.environ.get("NOTION_PAGE_ID")
        if not api_key or not page_id:
            return "Error: Missing Notion credentials (NOTION_API_KEY or NOTION_PAGE_ID)."

        notion = Client(auth=api_key)
        children = notion.blocks.children.list(block_id=page_id)
        ingredients = []
        for block in children.get("results", []):
            btype = block.get("type", "")
            rich_texts = block.get(btype, {}).get("rich_text", [])
            for rt in rich_texts:
                text = rt.get("plain_text", "").strip()
                if text:
                    for line in text.split("\n"):
                        line = line.strip().strip("-•* ")
                        if line:
                            ingredients.append(line)
        return json.dumps(ingredients, ensure_ascii=False)
    except Exception as e:
        return f"Error: {e}"


@tool
def search_recipes(ingredients: str) -> str:
    """Search the web for recipes using the given ingredients via Tavily.
    Input: comma-separated ingredient list.
    Returns a JSON list of up to 5 results with title, url, and content, or an error message starting with 'erro:'."""
    try:
        from tavily import TavilyClient

        api_key = os.environ.get("TAVILY_API_KEY")
        if not api_key:
            return "erro: Missing Tavily API key."

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
        return json.dumps(results, ensure_ascii=False)
    except Exception as e:
        return f"erro: {e}"


# --------------- State ---------------

class AgentState(TypedDict):
    query: str
    is_valid: bool
    validation_message: str
    ingredients: list[str]
    pantry_ingredients: list[str]
    search_results: list[dict]
    recipe: str
    error: str
    tool_messages: Annotated[list, lambda a, b: a + b]


# --------------- LLM ---------------

tools = [retrieve_pantry, search_recipes]
llm = ChatOpenAI(model="gpt-4o", temperature=0)


# --------------- Nodes ---------------

def validate_query(state: AgentState) -> dict:
    query = state["query"].strip() if state.get("query") else ""
    if not query:
        return {"is_valid": False, "validation_message": "Error: The query is empty."}

    response = llm.invoke(
        f"""You are a query classifier. Classify the following query as VALID or INVALID.
A query is INVALID if it is offensive, or completely unrelated to food/cooking/recipes.
A query is VALID if it relates in any way to food, cooking, ingredients, or recipes.

Query: "{query}"

Reply with exactly one word: VALID or INVALID"""
    )
    classification = response.content.strip().upper()
    if "INVALID" in classification:
        return {"is_valid": False, "validation_message": "Error: The query is not related to food or cooking, or is inappropriate."}
    return {"is_valid": True, "validation_message": ""}


def identify_ingredients(state: AgentState) -> dict:
    query = state["query"]
    response = llm.invoke(
        f"""You are a chef assistant. Analyze the following query and extract:
1. All ingredients explicitly mentioned as AVAILABLE (not negated, not hypothetical, not absent).
2. Any dietary restrictions mentioned (e.g., vegan, gluten-free, lactose intolerant, vegetarian, no pork, etc.).

IMPORTANT RULES:
- Do NOT include ingredients mentioned under negation ("no eggs", "without milk", "I don't have butter").
- Do NOT include ingredients mentioned hypothetically ("if I had chicken").
- Do NOT include ingredients mentioned as absent ("I'm out of flour").
- If a dietary restriction is declared, EXCLUDE all ingredients that violate it, even derived ones.
  For example: "vegan" excludes meat, fish, eggs, milk, butter, cheese, honey, gelatin, and ALL animal-derived products.
  "Gluten-free" excludes wheat, barley, rye, regular pasta, bread, flour (unless specified as gluten-free).
  "Vegetarian" excludes meat, fish, and seafood.

Query: "{query}"

Reply in JSON format:
{{"ingredients": ["ingredient1", "ingredient2"], "restrictions": ["restriction1"], "excluded_by_restriction": ["excluded1"]}}
If no ingredients found, return {{"ingredients": [], "restrictions": [], "excluded_by_restriction": []}}"""
    )
    try:
        content = response.content.strip()
        if "```json" in content:
            content = content.split("```json")[1].split("```")[0].strip()
        elif "```" in content:
            content = content.split("```")[1].split("```")[0].strip()
        data = json.loads(content)
        ingredients = data.get("ingredients", [])
        return {"ingredients": ingredients}
    except Exception:
        return {"ingredients": []}


def route_after_ingredients(state: AgentState) -> str:
    if state.get("ingredients"):
        return "search_recipes_node"
    return "pantry_lookup_node"


def search_recipes_node(state: AgentState) -> dict:
    ingredients_str = ", ".join(state["ingredients"])
    result = search_recipes.invoke({"ingredients": ingredients_str})
    if isinstance(result, str) and result.startswith("erro:"):
        return {"error": f"Error: Recipe search failed. {result}", "search_results": []}
    try:
        results = json.loads(result) if isinstance(result, str) else result
        return {"search_results": results, "error": ""}
    except Exception:
        return {"search_results": [], "error": ""}


def pantry_lookup_node(state: AgentState) -> dict:
    result = retrieve_pantry.invoke({})
    if isinstance(result, str) and result.startswith("Error:"):
        return {"error": result, "pantry_ingredients": []}
    try:
        pantry = json.loads(result) if isinstance(result, str) else result
        if not pantry:
            return {"pantry_ingredients": [], "error": ""}

        query = state["query"]
        response = llm.invoke(
            f"""Given the user's query: "{query}"
And the pantry ingredients: {json.dumps(pantry, ensure_ascii=False)}

Filter out any pantry ingredients that violate the dietary restrictions mentioned in the query.
Return ONLY the compatible ingredients as a JSON list.
If no restrictions are mentioned, return all pantry ingredients.

IMPORTANT: A dietary restriction implies ALL derived ingredients:
- "vegan" excludes ALL animal products (meat, fish, eggs, milk, butter, cheese, honey, gelatin, etc.)
- "gluten-free" excludes wheat, barley, rye, and products made from them
- "vegetarian" excludes meat, fish, and seafood

Reply with only a JSON list: ["ingredient1", "ingredient2"]"""
        )
        content = response.content.strip()
        if "```json" in content:
            content = content.split("```json")[1].split("```")[0].strip()
        elif "```" in content:
            content = content.split("```")[1].split("```")[0].strip()
        filtered = json.loads(content)
        return {"pantry_ingredients": filtered, "error": ""}
    except Exception as e:
        return {"pantry_ingredients": [], "error": f"Error: Failed to process pantry data. {e}"}


def route_after_pantry(state: AgentState) -> str:
    if state.get("error"):
        return "error_node"
    if not state.get("pantry_ingredients"):
        return "no_ingredients_node"
    return "generate_recipe_node"


def route_after_search(state: AgentState) -> str:
    if state.get("error"):
        return "error_node"
    return "generate_recipe_node"


def no_ingredients_node(state: AgentState) -> dict:
    return {"error": "Error: No compatible ingredients found in the pantry for the given dietary restrictions."}


def error_node(state: AgentState) -> dict:
    return {"error": state.get("error", "Error: An unexpected failure occurred.")}


def generate_recipe_node(state: AgentState) -> dict:
    query = state["query"]
    ingredients = state.get("ingredients", [])
    pantry_ingredients = state.get("pantry_ingredients", [])
    search_results = state.get("search_results", [])

    available = ingredients if ingredients else pantry_ingredients
    references = ""
    if search_results:
        references = "\n\nReference recipes found online (use as inspiration, not as the final recipe):\n"
        for r in search_results:
            references += f"- {r.get('title', '')}: {r.get('content', '')[:200]}\n"

    try:
        response = llm.invoke(
            f"""You are a professional chef creating a personalized recipe.

User request: "{query}"
Available ingredients: {json.dumps(available, ensure_ascii=False)}
{references}

IMPORTANT RULES:
- Respect ALL dietary restrictions mentioned in the query.
- Do NOT use any ingredient that violates the dietary restrictions, even if available.
- A dietary restriction implies ALL derived ingredients (e.g., vegan = no animal products at all).
- Use at least one of the available ingredients.
- You may add common pantry staples (salt, pepper, oil, water) if needed.

Create a complete recipe with EXACTLY these sections:

**Title:** [Recipe name]

**Ingredients:**
- [ingredient 1 with quantity]
- [ingredient 2 with quantity]

**Equipment:**
- [equipment 1]
- [equipment 2]

**Instructions:**
1. [Step 1]
2. [Step 2]

**Storage:**
- [Storage recommendation]

All sections must be present and non-empty."""
        )
        return {"recipe": response.content.strip()}
    except Exception as e:
        return {"error": f"Error: Failed to generate recipe. {e}"}


def format_output(state: AgentState) -> dict:
    return state


# --------------- Graph ---------------

def build_graph():
    graph = StateGraph(AgentState)

    graph.add_node("validate_query", validate_query)
    graph.add_node("identify_ingredients", identify_ingredients)
    graph.add_node("search_recipes_node", search_recipes_node)
    graph.add_node("pantry_lookup_node", pantry_lookup_node)
    graph.add_node("no_ingredients_node", no_ingredients_node)
    graph.add_node("error_node", error_node)
    graph.add_node("generate_recipe_node", generate_recipe_node)

    graph.set_entry_point("validate_query")

    graph.add_conditional_edges(
        "validate_query",
        lambda s: "identify_ingredients" if s.get("is_valid") else END,
    )
    graph.add_conditional_edges("identify_ingredients", route_after_ingredients)
    graph.add_conditional_edges("search_recipes_node", route_after_search)
    graph.add_conditional_edges("pantry_lookup_node", route_after_pantry)
    graph.add_edge("no_ingredients_node", END)
    graph.add_edge("error_node", END)
    graph.add_edge("generate_recipe_node", END)

    return graph.compile()


app = build_graph()


# --------------- Entrypoint ---------------

def agent(query: str) -> str:
    try:
        initial_state: AgentState = {
            "query": query,
            "is_valid": False,
            "validation_message": "",
            "ingredients": [],
            "pantry_ingredients": [],
            "search_results": [],
            "recipe": "",
            "error": "",
            "tool_messages": [],
        }
        result = app.invoke(initial_state)

        if result.get("error"):
            return result["error"]
        if result.get("validation_message"):
            return result["validation_message"]
        if result.get("recipe"):
            return result["recipe"]
        return "Error: No recipe was generated."
    except Exception as e:
        return f"Error: {e}"

import os
import json
from typing import TypedDict, Annotated, Any

from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage, ToolMessage
from langgraph.graph import StateGraph, END
import requests


# ── Tools ──

@tool
def retrieve_pantry() -> str:
    """Obtain the ingredients available in the user's pantry from a Notion page. Returns a JSON list of ingredient strings."""
    try:
        notion_api_key = os.environ.get("NOTION_API_KEY")
        page_id = os.environ.get("NOTION_PAGE_ID")
        if not notion_api_key or not page_id:
            return "Error: Missing Notion credentials (NOTION_API_KEY or NOTION_PAGE_ID)."
        headers = {
            "Authorization": f"Bearer {notion_api_key}",
            "Notion-Version": "2022-06-28",
        }
        url = f"https://api.notion.com/v1/blocks/{page_id}/children?page_size=100"
        resp = requests.get(url, headers=headers, timeout=30)
        if resp.status_code != 200:
            return f"Error: Notion API returned status {resp.status_code}."
        data = resp.json()
        ingredients = []
        for block in data.get("results", []):
            btype = block.get("type", "")
            rich_texts = block.get(btype, {}).get("rich_text", [])
            for rt in rich_texts:
                text = rt.get("plain_text", "").strip()
                if text:
                    ingredients.append(text)
        return json.dumps(ingredients, ensure_ascii=False)
    except requests.exceptions.Timeout:
        return "Error: Timeout while accessing the Notion page."
    except Exception as e:
        return f"Error: {e}"


@tool
def search_recipes(ingredients: str) -> str:
    """Search for recipes on the web using Tavily, as a preparation reference for the given ingredients. Input is a comma-separated list of ingredients. Returns a JSON list of up to 5 results with title, url, and content."""
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
                "content": r.get("content", ""),
            })
        return json.dumps(results, ensure_ascii=False)
    except Exception as e:
        return f"erro: {e}"


TOOLS = [retrieve_pantry, search_recipes]
TOOL_MAP = {t.name: t for t in TOOLS}

# ── State ──

class AgentState(TypedDict):
    query: str
    is_valid: bool
    error_message: str
    ingredients: list[str]
    pantry_ingredients: list[str]
    search_results: list[dict]
    recipe: str
    used_pantry: bool
    done: bool


# ── LLM ──

def get_llm():
    return ChatOpenAI(model="gpt-4o", temperature=0.3)


# ── Node functions ──

SYSTEM_PROMPT = (
    "You are a professional chef specialized in creating personalized recipes. "
    "You adapt recipes to available ingredients and respect dietary restrictions. "
    "You always respond in a structured format."
)


def validate_query(state: AgentState) -> AgentState:
    query = state["query"].strip() if state.get("query") else ""
    if not query:
        return {**state, "is_valid": False, "error_message": "Error: The query is empty.", "done": True}

    llm = get_llm()
    resp = llm.invoke([
        SystemMessage(content=SYSTEM_PROMPT),
        HumanMessage(content=(
            f"Classify the following query as 'valid' or 'invalid'. "
            f"A query is invalid if it is offensive, or completely unrelated to food/cooking/recipes. "
            f"A query about ingredients, meals, cooking, dietary preferences, or recipes is valid. "
            f"Respond with ONLY a JSON object: {{\"valid\": true/false, \"reason\": \"...\"}}\n\n"
            f"Query: \"{query}\""
        )),
    ])
    try:
        text = resp.content.strip()
        if text.startswith("```"):
            text = text.split("\n", 1)[1].rsplit("```", 1)[0].strip()
        result = json.loads(text)
        if not result.get("valid", True):
            return {**state, "is_valid": False, "error_message": f"Error: {result.get('reason', 'Invalid query.')}", "done": True}
    except Exception:
        pass
    return {**state, "is_valid": True, "error_message": "", "done": False}


def identify_ingredients(state: AgentState) -> AgentState:
    if not state.get("is_valid"):
        return state

    llm = get_llm()
    resp = llm.invoke([
        SystemMessage(content=SYSTEM_PROMPT),
        HumanMessage(content=(
            f"From the following user query, extract:\n"
            f"1. All food ingredients explicitly mentioned as AVAILABLE (not negated, not absent, not hypothetical).\n"
            f"2. Any dietary restrictions mentioned (e.g., vegan, gluten-free, lactose intolerant, vegetarian, no pork, etc.).\n\n"
            f"Then remove from the ingredient list any ingredient that violates the dietary restrictions, "
            f"including derived ingredients. For example, if the user is lactose intolerant, remove milk, cheese, butter, cream, yogurt, etc. "
            f"If vegan, remove all animal products. If vegetarian, remove meat and fish.\n\n"
            f"An ingredient mentioned under negation (e.g., 'no eggs', 'without milk', 'I don't have sugar') "
            f"must NOT be included.\n\n"
            f"Respond with ONLY a JSON object:\n"
            f"{{\"ingredients\": [\"ingredient1\", ...], \"restrictions\": [\"restriction1\", ...]}}\n\n"
            f"Query: \"{state['query']}\""
        )),
    ])
    try:
        text = resp.content.strip()
        if text.startswith("```"):
            text = text.split("\n", 1)[1].rsplit("```", 1)[0].strip()
        result = json.loads(text)
        ingredients = result.get("ingredients", [])
    except Exception:
        ingredients = []

    return {**state, "ingredients": ingredients}


def should_search_or_pantry(state: AgentState) -> str:
    if state.get("done"):
        return "end"
    if state.get("ingredients"):
        return "search_recipes"
    return "retrieve_pantry"


def do_search_recipes(state: AgentState) -> AgentState:
    try:
        ingredients_str = ", ".join(state["ingredients"])
        result = search_recipes.invoke({"ingredients": ingredients_str})
        if isinstance(result, str) and result.startswith("erro:"):
            return {**state, "error_message": f"Error: {result}", "done": True, "search_results": []}
        results = json.loads(result) if isinstance(result, str) else result
        return {**state, "search_results": results, "used_pantry": False}
    except Exception as e:
        return {**state, "error_message": f"Error: {e}", "done": True, "search_results": []}


def do_retrieve_pantry(state: AgentState) -> AgentState:
    try:
        result = retrieve_pantry.invoke({})
        if isinstance(result, str) and result.startswith("Error:"):
            return {**state, "error_message": f"Error: {result}", "done": True, "pantry_ingredients": []}
        pantry = json.loads(result) if isinstance(result, str) else result
        if not pantry:
            return {**state, "pantry_ingredients": [], "error_message": "Error: No ingredients found in the pantry compatible with the dietary restrictions.", "done": True}

        # Filter pantry by dietary restrictions
        llm = get_llm()
        resp = llm.invoke([
            SystemMessage(content=SYSTEM_PROMPT),
            HumanMessage(content=(
                f"Given the user query: \"{state['query']}\"\n"
                f"And the following pantry ingredients: {json.dumps(pantry)}\n\n"
                f"Remove any ingredient that violates dietary restrictions mentioned in the query "
                f"(including derived ingredients). Return ONLY a JSON array of compatible ingredients.\n"
                f"If no ingredient is compatible, return an empty array []."
            )),
        ])
        try:
            text = resp.content.strip()
            if text.startswith("```"):
                text = text.split("\n", 1)[1].rsplit("```", 1)[0].strip()
            filtered = json.loads(text)
        except Exception:
            filtered = pantry

        if not filtered:
            return {**state, "pantry_ingredients": [], "error_message": "Error: No ingredients found in the pantry compatible with the dietary restrictions.", "done": True}

        return {**state, "pantry_ingredients": filtered, "used_pantry": True}
    except Exception as e:
        return {**state, "error_message": f"Error: {e}", "done": True, "pantry_ingredients": []}


def should_generate_or_end(state: AgentState) -> str:
    if state.get("done"):
        return "end"
    return "generate_recipe"


def generate_recipe(state: AgentState) -> AgentState:
    llm = get_llm()

    if state.get("used_pantry"):
        ingredients_info = f"Available pantry ingredients: {json.dumps(state.get('pantry_ingredients', []))}"
        reference_info = ""
    else:
        ingredients_info = f"Available ingredients: {json.dumps(state.get('ingredients', []))}"
        refs = state.get("search_results", [])
        if refs:
            reference_info = "Reference recipes found online:\n" + "\n".join(
                f"- {r.get('title', '')}: {r.get('content', '')[:200]}" for r in refs
            )
        else:
            reference_info = ""

    resp = llm.invoke([
        SystemMessage(content=SYSTEM_PROMPT),
        HumanMessage(content=(
            f"User query: \"{state['query']}\"\n\n"
            f"{ingredients_info}\n"
            f"{reference_info}\n\n"
            f"Create a complete recipe using these ingredients that respects any dietary restrictions in the query. "
            f"You MUST use at least one of the provided ingredients. "
            f"Do NOT use any ingredient that violates the dietary restrictions.\n\n"
            f"Format the recipe EXACTLY as follows:\n"
            f"**Title:** <recipe title>\n\n"
            f"**Ingredients:**\n- <ingredient 1 with quantity>\n- <ingredient 2 with quantity>\n...\n\n"
            f"**Equipment:**\n- <equipment 1>\n- <equipment 2>\n...\n\n"
            f"**Instructions:**\n1. <step 1>\n2. <step 2>\n...\n\n"
            f"**Storage:**\n<storage recommendations>"
        )),
    ])
    return {**state, "recipe": resp.content, "done": True}


# ── Graph ──

def build_graph():
    g = StateGraph(AgentState)

    g.add_node("validate_query", validate_query)
    g.add_node("identify_ingredients", identify_ingredients)
    g.add_node("search_recipes_node", do_search_recipes)
    g.add_node("retrieve_pantry_node", do_retrieve_pantry)
    g.add_node("generate_recipe", generate_recipe)

    g.set_entry_point("validate_query")
    g.add_edge("validate_query", "identify_ingredients")
    g.add_conditional_edges("identify_ingredients", should_search_or_pantry, {
        "search_recipes": "search_recipes_node",
        "retrieve_pantry": "retrieve_pantry_node",
        "end": END,
    })
    g.add_conditional_edges("search_recipes_node", should_generate_or_end, {
        "generate_recipe": "generate_recipe",
        "end": END,
    })
    g.add_conditional_edges("retrieve_pantry_node", should_generate_or_end, {
        "generate_recipe": "generate_recipe",
        "end": END,
    })
    g.add_edge("generate_recipe", END)

    return g.compile()


# ── Entrypoint ──

def agent(query: str) -> str:
    try:
        graph = build_graph()
        initial_state: AgentState = {
            "query": query,
            "is_valid": False,
            "error_message": "",
            "ingredients": [],
            "pantry_ingredients": [],
            "search_results": [],
            "recipe": "",
            "used_pantry": False,
            "done": False,
        }
        result = graph.invoke(initial_state)
        if result.get("error_message"):
            return result["error_message"]
        return result.get("recipe", "Error: No recipe was generated.")
    except Exception as e:
        return f"Error: {e}"

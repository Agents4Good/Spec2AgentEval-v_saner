import os
from typing import TypedDict, Annotated, Any
from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage, ToolMessage
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode


# --- Tools ---

@tool
def retrieve_pantry() -> str:
    """Obtain the ingredients available in the user's pantry from a Notion page. Returns a list of ingredients or an empty list."""
    try:
        from notion_client import Client

        api_key = os.environ.get("NOTION_API_KEY")
        page_id = os.environ.get("NOTION_PAGE_ID")
        if not api_key or not page_id:
            return "Error: Missing Notion credentials (NOTION_API_KEY or NOTION_PAGE_ID)."

        notion = Client(auth=api_key)
        blocks = notion.blocks.children.list(block_id=page_id)
        ingredients = []
        for block in blocks.get("results", []):
            btype = block.get("type", "")
            if btype in ("bulleted_list_item", "numbered_list_item", "paragraph"):
                rich = block.get(btype, {}).get("rich_text", [])
                text = "".join(r.get("plain_text", "") for r in rich).strip()
                if text:
                    ingredients.append(text)
        return str(ingredients) if ingredients else "[]"
    except Exception as e:
        return f"Error: {e}"


@tool
def search_recipes(ingredients: str) -> str:
    """Search for recipes on the web using Tavily based on the given ingredients. Returns up to 5 results with title, URL, and content excerpt."""
    try:
        api_key = os.environ.get("TAVILY_API_KEY")
        if not api_key:
            return "erro: Missing Tavily API key."

        from tavily import TavilyClient
        client = TavilyClient(api_key=api_key)
        query = f"recipe with {ingredients}"
        results = client.search(query=query, max_results=5)
        output = []
        for r in results.get("results", []):
            output.append({
                "title": r.get("title", ""),
                "url": r.get("url", ""),
                "content": r.get("content", "")[:500]
            })
        return str(output) if output else "[]"
    except Exception as e:
        return f"erro: {e}"


# --- State ---

class AgentState(TypedDict):
    query: str
    is_valid: bool
    error_message: str
    ingredients: list[str]
    pantry_ingredients: list[str]
    search_results: str
    recipe: str
    messages: list[Any]
    tools_called: list[str]
    current_step: str


# --- LLM ---

def get_llm():
    return ChatOpenAI(model="gpt-4o", temperature=0.3)


# --- Node functions ---

def validate_query(state: AgentState) -> dict:
    query = state["query"].strip()
    if not query:
        return {"is_valid": False, "error_message": "Error: The query is empty.", "current_step": "done"}

    llm = get_llm()
    resp = llm.invoke([
        SystemMessage(content="You are a query validator. Classify the query as 'valid' or 'invalid'. A query is invalid if it is offensive, or completely unrelated to food/cooking/recipes/ingredients. Respond with ONLY 'valid' or 'invalid'."),
        HumanMessage(content=f"Query: {query}")
    ])
    result = resp.content.strip().lower()
    if "invalid" in result:
        return {"is_valid": False, "error_message": "Error: The query is not related to food or cooking, or is inappropriate.", "current_step": "done"}
    return {"is_valid": True, "current_step": "identify_ingredients"}


def identify_ingredients(state: AgentState) -> dict:
    query = state["query"]
    llm = get_llm()
    resp = llm.invoke([
        SystemMessage(content="""You are a culinary ingredient identifier. Given a user query about food/recipes:
1. Identify all ingredients explicitly mentioned as AVAILABLE.
2. Do NOT include ingredients mentioned under negation, absence, or hypothesis (e.g., "without X", "no X", "if I had X").
3. Identify any dietary restrictions (e.g., vegan, gluten-free, vegetarian, lactose-free, nut-free, etc.).
4. Remove any ingredient that violates the dietary restrictions, including derived ingredients. For example, if vegan, remove all animal products (meat, dairy, eggs, honey, etc.); if gluten-free, remove wheat, barley, rye, and anything derived from them.

Respond in this exact format:
INGREDIENTS: comma-separated list or NONE
RESTRICTIONS: comma-separated list or NONE"""),
        HumanMessage(content=query)
    ])

    text = resp.content.strip()
    ingredients = []
    restrictions_line = ""

    for line in text.split("\n"):
        line = line.strip()
        if line.upper().startswith("INGREDIENTS:"):
            val = line.split(":", 1)[1].strip()
            if val.upper() != "NONE" and val:
                ingredients = [i.strip() for i in val.split(",") if i.strip()]
        elif line.upper().startswith("RESTRICTIONS:"):
            restrictions_line = line.split(":", 1)[1].strip()

    return {"ingredients": ingredients, "current_step": "route_after_ingredients"}


def route_after_ingredients(state: AgentState) -> str:
    if state.get("ingredients"):
        return "search_recipes_node"
    return "pantry_lookup_node"


def search_recipes_node(state: AgentState) -> dict:
    tools = [search_recipes]
    tool_node = ToolNode(tools)

    ingredients_str = ", ".join(state["ingredients"])

    llm = get_llm().bind_tools(tools)
    resp = llm.invoke([
        SystemMessage(content="You must search for recipes using the search_recipes tool. Call it with the ingredients provided."),
        HumanMessage(content=f"Search for recipes with these ingredients: {ingredients_str}")
    ])

    if resp.tool_calls:
        tool_messages = tool_node.invoke({"messages": [resp]})
        result_messages = tool_messages.get("messages", [])
        result_content = ""
        for msg in result_messages:
            if isinstance(msg, ToolMessage):
                result_content = msg.content

        if result_content.startswith("erro:"):
            return {
                "error_message": f"Error: {result_content}",
                "search_results": "",
                "tools_called": state.get("tools_called", []) + ["search_recipes"],
                "current_step": "done"
            }

        return {
            "search_results": result_content,
            "tools_called": state.get("tools_called", []) + ["search_recipes"],
            "current_step": "generate_recipe"
        }

    return {"search_results": "[]", "tools_called": state.get("tools_called", []) + ["search_recipes"], "current_step": "generate_recipe"}


def pantry_lookup_node(state: AgentState) -> dict:
    tools = [retrieve_pantry]
    tool_node = ToolNode(tools)

    llm = get_llm().bind_tools(tools)
    resp = llm.invoke([
        SystemMessage(content="You must retrieve the user's pantry using the retrieve_pantry tool."),
        HumanMessage(content="Retrieve the pantry ingredients.")
    ])

    if resp.tool_calls:
        tool_messages = tool_node.invoke({"messages": [resp]})
        result_messages = tool_messages.get("messages", [])
        result_content = ""
        for msg in result_messages:
            if isinstance(msg, ToolMessage):
                result_content = msg.content

        if result_content.startswith("Error:"):
            return {
                "error_message": f"Error: {result_content}",
                "pantry_ingredients": [],
                "tools_called": state.get("tools_called", []) + ["retrieve_pantry"],
                "current_step": "done"
            }

        # Filter pantry ingredients by dietary restrictions
        pantry_raw = result_content
        query = state["query"]

        filter_llm = get_llm()
        filter_resp = filter_llm.invoke([
            SystemMessage(content="""Given a list of pantry ingredients and dietary restrictions from the user query, return only the ingredients that are compatible with those restrictions. A dietary restriction implies all derived ingredients must also be excluded.

Respond with ONLY a comma-separated list of compatible ingredients, or NONE if no ingredient is compatible."""),
            HumanMessage(content=f"Pantry ingredients: {pantry_raw}\nUser query: {query}")
        ])

        filtered = filter_resp.content.strip()
        if filtered.upper() == "NONE" or not filtered:
            return {
                "pantry_ingredients": [],
                "tools_called": state.get("tools_called", []) + ["retrieve_pantry"],
                "current_step": "check_pantry_empty"
            }

        pantry_list = [i.strip() for i in filtered.split(",") if i.strip()]
        return {
            "pantry_ingredients": pantry_list,
            "tools_called": state.get("tools_called", []) + ["retrieve_pantry"],
            "current_step": "check_pantry_empty"
        }

    return {
        "pantry_ingredients": [],
        "tools_called": state.get("tools_called", []) + ["retrieve_pantry"],
        "current_step": "check_pantry_empty"
    }


def route_after_pantry(state: AgentState) -> str:
    if state.get("pantry_ingredients"):
        return "generate_recipe"
    return "no_ingredients"


def no_ingredients(state: AgentState) -> dict:
    return {
        "error_message": "Error: No compatible ingredients found in your pantry for the given dietary restrictions.",
        "current_step": "done"
    }


def generate_recipe(state: AgentState) -> dict:
    query = state["query"]
    ingredients = state.get("ingredients", [])
    pantry = state.get("pantry_ingredients", [])
    search_results = state.get("search_results", "")

    available = ingredients if ingredients else pantry
    source = "query" if ingredients else "pantry"

    llm = get_llm()
    prompt = f"""You are a professional chef creating a personalized recipe.

User query: {query}
Available ingredients ({source}): {', '.join(available)}
Reference recipes from web search: {search_results if search_results else 'None available'}

Create a complete recipe using the available ingredients. The recipe MUST:
- Respect any dietary restrictions mentioned in the query
- Use at least one of the available ingredients listed above
- NOT include any ingredient that violates dietary restrictions

Respond in this EXACT format:

Title: [recipe title]

Ingredients:
- [ingredient 1 with quantity]
- [ingredient 2 with quantity]

Equipment:
- [equipment 1]
- [equipment 2]

Instructions:
1. [step 1]
2. [step 2]

Storage:
- [storage recommendation 1]"""

    resp = llm.invoke([HumanMessage(content=prompt)])
    return {"recipe": resp.content.strip(), "current_step": "done"}


def format_output(state: AgentState) -> dict:
    if state.get("error_message"):
        return {"recipe": state["error_message"]}
    return {}


# --- Graph ---

def build_graph():
    graph = StateGraph(AgentState)

    graph.add_node("validate_query", validate_query)
    graph.add_node("identify_ingredients", identify_ingredients)
    graph.add_node("search_recipes_node", search_recipes_node)
    graph.add_node("pantry_lookup_node", pantry_lookup_node)
    graph.add_node("no_ingredients", no_ingredients)
    graph.add_node("generate_recipe", generate_recipe)
    graph.add_node("format_output", format_output)

    graph.set_entry_point("validate_query")

    graph.add_conditional_edges("validate_query", lambda s: "identify_ingredients" if s["is_valid"] else "format_output")
    graph.add_conditional_edges("identify_ingredients", route_after_ingredients)
    graph.add_conditional_edges("search_recipes_node", lambda s: "generate_recipe" if not s.get("error_message") else "format_output")
    graph.add_conditional_edges("pantry_lookup_node", route_after_pantry)
    graph.add_edge("no_ingredients", "format_output")
    graph.add_edge("generate_recipe", "format_output")
    graph.add_edge("format_output", END)

    return graph.compile()


def agent(query: str) -> str:
    try:
        app = build_graph()
        initial_state: AgentState = {
            "query": query,
            "is_valid": False,
            "error_message": "",
            "ingredients": [],
            "pantry_ingredients": [],
            "search_results": "",
            "recipe": "",
            "messages": [],
            "tools_called": [],
            "current_step": "start"
        }
        result = app.invoke(initial_state)
        if result.get("error_message"):
            return result["error_message"]
        return result.get("recipe", "Error: Failed to generate recipe.")
    except Exception as e:
        return f"Error: {e}"
